"""CyclicGen_train.py"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dataset
import numpy as np
import os
import tensorflow as tf
from datetime import datetime
from utils.image_utils import imwrite
from skimage.measure import compare_ssim as ssim
from vgg16 import Vgg16
from pathlib import Path
from logging import getLogger
import logging


logger = getLogger(__name__)

FLAGS = tf.app.flags.FLAGS

# Define necessary FLAGS
tf.app.flags.DEFINE_string('train_dir', './train_dir',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_string('subset', 'train',
                           """Either 'train' or 'test'.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', None,
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")
tf.app.flags.DEFINE_integer('max_steps', 10000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('batch_size', 8, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer('training_data_step', 1, """The step used to reduce training data size""")
tf.app.flags.DEFINE_string('model_size', 'large', """The size of model""") ##
tf.app.flags.DEFINE_string('dataset', 'ucf101_256', """dataset (ucf101_256 or middlebury) """) ##
tf.app.flags.DEFINE_string('stage', 's1s2', """stage (s1 or s2)""") ##
tf.app.flags.DEFINE_integer('s1_steps', 10000, """ number of steps for stage1 if 's1s2' is specified as stage. """) ##
tf.app.flags.DEFINE_integer('logging_interval', 10, """ number of steps of interval to log. """) ##
tf.app.flags.DEFINE_integer('checkpoint_interval', 5000, """ number of steps of interval to save checkpoints. """) ##
tf.app.flags.DEFINE_integer('save_summary', 0, """ save summary if 1.  """) ##


def _read_image(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string, channels=3)
    # image_decoded.set_shape([256, 256, 3])
    return tf.cast(image_decoded, dtype=tf.float32) / 127.5 - 1.0

"""
def random_scaling(image, seed=1):
    scaling = tf.random_uniform([], 0.4, 0.6, seed=seed)
    return tf.image.resize_images(image, [tf.cast(tf.round(256*scaling), tf.int32), tf.cast(tf.round(256*scaling), tf.int32)])
"""

def train(dataset_frame1, dataset_frame2, dataset_frame3, out_dir, log_sep=' ,',hist_logger=None):
    """Trains a model."""
    graph = tf.Graph()
    with graph.as_default():
        s2_flag_tensor = tf.placeholder(dtype="float", shape=None)
        stage = FLAGS.stage

        # Create input.
        data_list_frame1 = dataset_frame1.read_data_list_file()
        data_list_frame1 = data_list_frame1[::FLAGS.training_data_step]
        dataset_frame1 = tf.data.Dataset.from_tensor_slices(tf.constant(data_list_frame1))
        dataset_frame1 = dataset_frame1.apply(
            tf.contrib.data.shuffle_and_repeat(buffer_size=1000000, count=None, seed=1)).map(_read_image).map(
            lambda image: tf.image.random_flip_left_right(image, seed=1)).map(
            lambda image: tf.image.random_flip_up_down(image, seed=1)).map(
            lambda image: tf.random_crop(image, [256, 256, 3], seed=1))
        dataset_frame1 = dataset_frame1.prefetch(8)

        data_list_frame2 = dataset_frame2.read_data_list_file()
        data_list_frame2 = data_list_frame2[::FLAGS.training_data_step]
        dataset_frame2 = tf.data.Dataset.from_tensor_slices(tf.constant(data_list_frame2))
        dataset_frame2 = dataset_frame2.apply(
            tf.contrib.data.shuffle_and_repeat(buffer_size=1000000, count=None, seed=1)).map(_read_image).map(
            lambda image: tf.image.random_flip_left_right(image, seed=1)).map(
            lambda image: tf.image.random_flip_up_down(image, seed=1)).map(
            lambda image: tf.random_crop(image, [256, 256, 3], seed=1))
        dataset_frame2 = dataset_frame2.prefetch(8)


        data_list_frame3 = dataset_frame3.read_data_list_file()
        data_list_frame3 = data_list_frame3[::FLAGS.training_data_step]
        dataset_frame3 = tf.data.Dataset.from_tensor_slices(tf.constant(data_list_frame3))
        dataset_frame3 = dataset_frame3.apply(
            tf.contrib.data.shuffle_and_repeat(buffer_size=1000000, count=None, seed=1)).map(_read_image).map(
            lambda image: tf.image.random_flip_left_right(image, seed=1)).map(
            lambda image: tf.image.random_flip_up_down(image, seed=1)).map(
            lambda image: tf.random_crop(image, [256, 256, 3], seed=1))
        dataset_frame3 = dataset_frame3.prefetch(8)

        batch_frame1 = dataset_frame1.batch(FLAGS.batch_size).make_initializable_iterator()
        batch_frame2 = dataset_frame2.batch(FLAGS.batch_size).make_initializable_iterator()
        batch_frame3 = dataset_frame3.batch(FLAGS.batch_size).make_initializable_iterator()

        # Create input and target placeholder.
        input1 = batch_frame1.get_next()
        input2 = batch_frame2.get_next()
        input3 = batch_frame3.get_next()


        edge_vgg_1 = Vgg16(input1,reuse=None)
        if True: #if stage == 's2':
            edge_vgg_2 = Vgg16(input2,reuse=True)
        edge_vgg_3 = Vgg16(input3,reuse=True)

        edge_1 = tf.nn.sigmoid(edge_vgg_1.fuse)
        if True: #if stage == 's2':
            edge_2 = tf.nn.sigmoid(edge_vgg_2.fuse)
        edge_3 = tf.nn.sigmoid(edge_vgg_3.fuse)

        edge_1 = tf.reshape(edge_1,[-1,input1.get_shape().as_list()[1],input1.get_shape().as_list()[2],1])
        if True: #if stage == 's2':
            edge_2 = tf.reshape(edge_2,[-1,input1.get_shape().as_list()[1],input1.get_shape().as_list()[2],1])
        edge_3 = tf.reshape(edge_3,[-1,input1.get_shape().as_list()[1],input1.get_shape().as_list()[2],1])

        if False: #if stage == 's1':
            with tf.variable_scope("Cycle_DVF"):
                model1_s1_i00_i20 = Voxel_flow_model()
                prediction1, flow1 = model1_s1_i00_i20.inference(tf.concat([input1, input3, edge_1, edge_3], 3))
                reconstruction_loss = model1_s1_i00_i20.l1loss(prediction1, input2)

        if True: #if stage == 's2':
            input_placeholder1 = tf.concat([input1, input2], 3)
            input_placeholder2 = tf.concat([input2, input3], 3)

            input_placeholder1 = tf.concat([input_placeholder1, edge_1, edge_2], 3)
            input_placeholder2 = tf.concat([input_placeholder2, edge_2, edge_3], 3)

            with tf.variable_scope("Cycle_DVF"):
                model1_s2_i00_i10 = Voxel_flow_model()
                prediction1, flow1 = model1_s2_i00_i10.inference(input_placeholder1)

            with tf.variable_scope("Cycle_DVF", reuse=True):
                model2_s2_i10_i20 = Voxel_flow_model()
                prediction2, flow2 = model2_s2_i10_i20.inference(input_placeholder2)

            edge_vgg_prediction1 = Vgg16(prediction1,reuse=True)
            edge_vgg_prediction2 = Vgg16(prediction2,reuse=True)

            edge_prediction1 = tf.nn.sigmoid(edge_vgg_prediction1.fuse)
            edge_prediction2 = tf.nn.sigmoid(edge_vgg_prediction2.fuse)

            edge_prediction1 = tf.reshape(edge_prediction1,[-1,input1.get_shape().as_list()[1],input1.get_shape().as_list()[2],1])
            edge_prediction2 = tf.reshape(edge_prediction2,[-1,input1.get_shape().as_list()[1],input1.get_shape().as_list()[2],1])

            with tf.variable_scope("Cycle_DVF", reuse=True):
                model3_s2_i05_i15 = Voxel_flow_model()
                prediction3, flow3 = model3_s2_i05_i15.inference(tf.concat([prediction1, prediction2, edge_prediction1, edge_prediction2], 3))
                cycle_consistency_loss = model3_s2_i05_i15.l1loss(prediction3, input2)

            with tf.variable_scope("Cycle_DVF", reuse=True):
                model4_s2_i00_i20 = Voxel_flow_model()
                prediction4, flow4 = model4_s2_i00_i20.inference(tf.concat([input1, input3,edge_1,edge_3], 3))
                reconstruction_loss = model4_s2_i00_i20.l1loss(prediction4, input2)

        t_vars = tf.trainable_variables()
        #logger.debug('all layers:')
        #for var in t_vars: logger.debug(var.name)
        logger.debug('all_layers:' + ' | '.join([var.name for var in t_vars]))
        dof_vars = [var for var in t_vars if not 'hed' in var.name]
        #logger.debug('optimize layers:')
        #for var in dof_vars: logger.debug(var.name)
        logger.debug('optimize layers:' + ' | '.join([var.name for var in dof_vars]))

        if False: #if stage == 's1':
            cycle_consistency_loss = tf.convert_to_tensor(0.0, dtype=tf.float32)
            motion_linearity_loss =  tf.convert_to_tensor(0.0, dtype=tf.float32)
            total_loss = reconstruction_loss

        if True: #if stage == 's2':
            motion_linearity_loss = tf.reduce_mean(tf.square(model4_s2_i00_i20.flow - model3_s2_i05_i15.flow * 2.0))
            total_loss = reconstruction_loss + s2_flag_tensor * cycle_consistency_loss + s2_flag_tensor * 0.1 * motion_linearity_loss

        # Perform learning rate scheduling.
        # Create an optimizer that performs gradient descent.

        learning_rate_s1 = 0.0001
        learning_rate_s2 = 0.00001
        with tf.variable_scope(tf.get_variable_scope(), reuse=None):
            update_op_s1 = tf.train.AdamOptimizer(learning_rate_s1).minimize(total_loss, var_list=dof_vars)
            #opt = tf.train.AdamOptimizer(learning_rate_s1)
            #update_op = opt.minimize(total_loss, var_list=dof_vars)
            update_op_s2 = tf.train.AdamOptimizer(learning_rate_s2).minimize(total_loss, var_list=dof_vars)

        init = tf.global_variables_initializer()  # init = tf.initialize_all_variables()

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=50) # saver = tf.train.Saver(tf.all_variables(), max_to_keep=50)

        if FLAGS.save_summary:
            # Create summaries
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
            summaries.append(tf.summary.scalar('total_loss', total_loss))
            summaries.append(tf.summary.image('input1', input1, 3))
            summaries.append(tf.summary.image('input2', input2, 3))
            summaries.append(tf.summary.image('input3', input3, 3))
            summaries.append(tf.summary.image('edge_1', edge_1, 3))
            if True: #if stage == 's2':
                summaries.append(tf.summary.image('edge_2', edge_1, 3))
            summaries.append(tf.summary.image('edge_3', edge_1, 3))

            if False: #if stage == 's1':
                summaries.append(tf.summary.image('prediction1', prediction1, 3))
            if True: #if stage == 's2':
                summaries.append(tf.summary.image('prediction3', prediction3, 3))
                summaries.append(tf.summary.image('prediction4', prediction4, 3))

            # Build the summary operation from the last tower summaries.
            summary_op = tf.summary.merge_all()

        with tf.Session(graph=graph) as sess:
            s2_flag = np.float32(0.0)
            update_op = update_op_s1
            learning_rate = learning_rate_s1

            last_step = -1

            # Restore checkpoint from file.
            if FLAGS.pretrained_model_checkpoint_path:
                restorer = tf.train.Saver()
                restorer.restore(sess, FLAGS.pretrained_model_checkpoint_path)
                logger.info('%s: Pre-trained model restored from %s' %
                      (datetime.now(), FLAGS.pretrained_model_checkpoint_path))
                try:
                    last_step = int(str(FLAGS.pretrained_model_checkpoint_path).split(sep='-')[-1])
                except ValueError:
                    logger.warning('The step number could not retrieved from the checkpoint path.'
                          'Continue running.')

            else:
                # Build an initialization operation to run below.

                sess.run([init], feed_dict={s2_flag_tensor: s2_flag})

            sess.run([batch_frame1.initializer, batch_frame2.initializer, batch_frame3.initializer], feed_dict={s2_flag_tensor: s2_flag})

            meta_model_file = 'hed_model/new-model.ckpt'
            saver2 = tf.train.Saver(var_list=[v for v in tf.global_variables() if "hed" in v.name])
            #saver2 = tf.train.Saver(var_list=[v for v in tf.all_variables() if "hed" in v.name])
            saver2.restore(sess, meta_model_file)

            if FLAGS.save_summary:
                # Summary Writter
                summary_writer = tf.summary.FileWriter(
                    out_dir,
                    graph=sess.graph)

            data_size = len(data_list_frame1)
            num_batches_per_epoch = int(data_size / FLAGS.batch_size)

            if FLAGS.stage == 's1s2':
                s1_steps = FLAGS.s1_steps
            if FLAGS.stage == 's2':
                s1_steps = 0
            if FLAGS.stage == 's1':
                s1_steps = FLAGS.max_steps

            initial_step = last_step + 1

            total_loss_ssum = 0
            reconstruction_loss_ssum = 0
            cycle_consistency_loss_ssum = 0
            motion_linearity_loss_ssum = 0

            for step_i in range(initial_step, FLAGS.max_steps):
                batch_idx = step_i % num_batches_per_epoch

                # Run single step update.
                if step_i == s1_steps:
                    s2_flag = np.float32(1.0)
                    update_op = update_op_s2
                    learning_rate = learning_rate_s2
                #if step_i == s1_steps:
                    #s2_flag = np.float32(1.0)
                    #learning_rate_s1 = 0.00001
                #if step_i in [initial_step, s1_steps]:

                sess.run(update_op, feed_dict={s2_flag_tensor: s2_flag})

                if batch_idx == 0:
                    logger.info('Epoch Number: %d' % int(step_i // num_batches_per_epoch))

                if True: # if step_i % FLAGS.logging_interval == 0:
                    total_loss_bsum, reconstruction_loss_bsum, cycle_consistency_loss_bsum, motion_linearity_loss_bsum = \
                        sess.run([total_loss,
                                  reconstruction_loss, cycle_consistency_loss, motion_linearity_loss],
                                 feed_dict={s2_flag_tensor: s2_flag})

                    total_loss_ssum += total_loss_bsum
                    reconstruction_loss_ssum += reconstruction_loss_bsum
                    cycle_consistency_loss_ssum += cycle_consistency_loss_bsum
                    motion_linearity_loss_ssum += motion_linearity_loss_bsum

                if step_i % FLAGS.logging_interval == (FLAGS.logging_interval-1):
                    total_loss_mean = total_loss_ssum / (FLAGS.logging_interval * FLAGS.batch_size)
                    reconstruction_loss_mean = reconstruction_loss_ssum / (FLAGS.logging_interval * FLAGS.batch_size)
                    cycle_consistency_loss_mean = cycle_consistency_loss_ssum / (FLAGS.logging_interval * FLAGS.batch_size)
                    motion_linearity_loss_mean = motion_linearity_loss_ssum / (FLAGS.logging_interval * FLAGS.batch_size)

                    hist_latest_str = log_sep.join(['Hist', '{:06d}', '{:.9e}', '{:.9e}', '{:.9e}', '{:.9e}', '{:.9e}']).format( \
                        step_i,
                        learning_rate,
                        total_loss_mean,
                        reconstruction_loss_mean,
                        cycle_consistency_loss_mean,
                        motion_linearity_loss_mean)

                    if hist_logger is None:
                        logger.info(hist_latest_str)

                    if hist_logger is not None:
                        hist_logger(step_i,
                                   learning_rate,
                                   total_loss_mean,
                                   reconstruction_loss_mean,
                                   cycle_consistency_loss_mean,
                                   motion_linearity_loss_mean)
                        print(hist_latest_str)

                    total_loss_ssum = 0
                    reconstruction_loss_ssum = 0
                    cycle_consistency_loss_ssum = 0
                    motion_linearity_loss_ssum = 0

                # Save checkpoint
                if step_i % FLAGS.checkpoint_interval == (FLAGS.checkpoint_interval-1) or ((step_i) == (FLAGS.max_steps-1)):
                    # Output Summary
                    if FLAGS.save_summary:
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, step_i)
                    #
                    checkpoint_path = os.path.join(out_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step_i)

    sess.close()

def validate(dataset_frame1, dataset_frame2, dataset_frame3):
    """Performs validation on model.
    Args:
    """
    pass


def test(dataset_frame1, dataset_frame2, dataset_frame3):
    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    """Perform test on a trained model."""
    with tf.Graph().as_default():
        # Create input and target placeholder.
        input_placeholder = tf.placeholder(tf.float32, shape=(None, 256, 256, 6))
        target_placeholder = tf.placeholder(tf.float32, shape=(None, 256, 256, 3))

        edge_vgg_1 = Vgg16(input_placeholder[:, :, :, :3], reuse=None)
        edge_vgg_3 = Vgg16(input_placeholder[:, :, :, 3:6], reuse=True)

        edge_1 = tf.nn.sigmoid(edge_vgg_1.fuse)
        edge_3 = tf.nn.sigmoid(edge_vgg_3.fuse)

        edge_1 = tf.reshape(edge_1, [-1, input_placeholder.get_shape().as_list()[1], input_placeholder.get_shape().as_list()[2], 1])
        edge_3 = tf.reshape(edge_3, [-1, input_placeholder.get_shape().as_list()[1], input_placeholder.get_shape().as_list()[2], 1])

        with tf.variable_scope("Cycle_DVF"):
            # Prepare model.
            model = Voxel_flow_model(is_train=False)
            prediction = model.inference(tf.concat([input_placeholder, edge_1, edge_3], 3))

        # Create a saver and load.
        sess = tf.Session()

        # Restore checkpoint from file.
        if FLAGS.pretrained_model_checkpoint_path:
            restorer = tf.train.Saver()
            restorer.restore(sess, FLAGS.pretrained_model_checkpoint_path)
            logger.info('%s: Pre-trained model restored from %s' %
                  (datetime.now(), FLAGS.pretrained_model_checkpoint_path))

        # Process on test dataset.
        data_list_frame1 = dataset_frame1.read_data_list_file()
        data_size = len(data_list_frame1)

        data_list_frame2 = dataset_frame2.read_data_list_file()

        data_list_frame3 = dataset_frame3.read_data_list_file()

        i = 0
        PSNR = 0
        SSIM = 0

        for id_img in range(0, data_size):
            UCF_index = data_list_frame1[id_img][:-12]
            # Load single data.

            batch_data_frame1 = [dataset_frame1.process_func(os.path.join('ucf101_interp_ours', ll)[:-5] + '00.png') for
                                 ll in data_list_frame1[id_img:id_img + 1]]
            batch_data_frame2 = [dataset_frame2.process_func(os.path.join('ucf101_interp_ours', ll)[:-5] + '01_gt.png')
                                 for ll in data_list_frame2[id_img:id_img + 1]]
            batch_data_frame3 = [dataset_frame3.process_func(os.path.join('ucf101_interp_ours', ll)[:-5] + '02.png') for
                                 ll in data_list_frame3[id_img:id_img + 1]]
            batch_data_mask = [
                dataset_frame3.process_func(os.path.join('motion_masks_ucf101_interp', ll)[:-11] + 'motion_mask.png')
                for ll in data_list_frame3[id_img:id_img + 1]]

            batch_data_frame1 = np.array(batch_data_frame1)
            batch_data_frame2 = np.array(batch_data_frame2)
            batch_data_frame3 = np.array(batch_data_frame3)
            batch_data_mask = (np.array(batch_data_mask) + 1.0) / 2.0

            feed_dict = {input_placeholder: np.concatenate((batch_data_frame1, batch_data_frame3), 3),
                         target_placeholder: batch_data_frame2}
            # Run single step update.
            prediction_np, target_np, warped_img1, warped_img2 = sess.run([prediction,
                                                                                       target_placeholder, model.warped_img1,
                                                                                       model.warped_img2],
                                                                                      feed_dict=feed_dict)

            imwrite('ucf101_interp_ours/' + str(UCF_index) + '/frame_01_CyclicGen.png', prediction_np[0][-1, :, :, :])

            logger.info(np.sum(batch_data_mask))
            if np.sum(batch_data_mask) > 0:
                img_pred_mask = np.expand_dims(batch_data_mask[0], -1) * (prediction_np[0][-1] + 1.0) / 2.0
                img_target_mask = np.expand_dims(batch_data_mask[0], -1) * (target_np[-1] + 1.0) / 2.0
                mse = np.sum((img_pred_mask - img_target_mask) ** 2) / (3. * np.sum(batch_data_mask))
                psnr_cur = 20.0 * np.log10(1.0) - 10.0 * np.log10(mse)

                img_pred_gray = rgb2gray((prediction_np[0][-1] + 1.0) / 2.0)
                img_target_gray = rgb2gray((target_np[-1] + 1.0) / 2.0)
                ssim_cur = ssim(img_pred_gray, img_target_gray, data_range=1.0)

                PSNR += psnr_cur
                SSIM += ssim_cur

                i += 1
        logger.info("Overall PSNR: %f db" % (PSNR / i))
        logger.info("Overall SSIM: %f db" % (SSIM / i))


hist_logging = False
try:
    from table_logger import TableLogger
    hist_logging = True
except:
    print('Continue running without logging to a CSV file as table-logger is not installed.'
          ' To enable logging, "pip install table-logger" and rerun this code.')


def timestamp():
    return datetime.now().strftime('%Y-%m-%dT%H%M%S')


if __name__ == '__main__':

    start_time = timestamp()

    config_str = '_'.join([FLAGS.model_size, FLAGS.dataset, FLAGS.stage, start_time])
    out_dir = FLAGS.train_dir + '/' + config_str
    Path(out_dir).mkdir(parents=True, exist_ok=True)



    log_sep = ' ,'

    hist_logger = None
    if hist_logging:
        history_cols = ['Step', 'Learning_Rate', 'Loss', 'Reconstruction_Loss', 'Cycle_Consistency_Loss',
                        'Motion_Linearity_Loss']
        file_name = out_dir + '/hist_{}.csv'.format(config_str)
        hist_logger = TableLogger(csv=True, file=file_name,
                                 columns=','.join(history_cols),
                                 rownum=True, time_delta=True, timestamp=True,
                                 float_format='{:.9e}'.format)
        logger.info('The loss values will be logged to: {}'.format(file_name))

    """
    format_str = log_sep.join([])
    logging.basicConfig(level=logging.DEBUG,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(log_file_path),
                            logging.StreamHandler()
                        ]
                        )
    history_cols = ['Step','Learning_Rate','Loss','Reconstruction_Loss','Cycle_Consistency_Loss','Motion_Linearity_Loss']
    logger.info(log_sep.join(['Datetime', 'Level', 'Hist'] + history_cols))
    """

    log_file_path = out_dir + '/log_{}.csv'.format(config_str)
    format_str = log_sep.join(['%(asctime)s.%(msecs)03d','%(module)s','%(funcName)s','%(levelname)s','%(message)s'])
    #format_str = log_sep.join(['%(asctime)s.%(msecs)03d', '%(levelname)s', '%(message)s'])
    logging.basicConfig(level=logging.DEBUG,
                        format=format_str, datefmt='%Y-%m-%dT%H:%M:%S',
                        handlers=[
                            logging.FileHandler(log_file_path),
                            logging.StreamHandler()
                        ]
                        )
    #history_cols = ['Step','Learning_Rate','Loss','Reconstruction_Loss','Cycle_Consistency_Loss','Motion_Linearity_Loss']
    #logger.info(log_sep.join(['Hist'] + history_cols))

    try:
        logger.info('train_dir: {}'.format(FLAGS.train_dir))
        logger.info('subset: {}'.format(FLAGS.subset))
        logger.info('pretrained_model_checkpoint_path: {}'.format(FLAGS.pretrained_model_checkpoint_path))
        logger.info('max_steps: {}'.format(FLAGS.max_steps))
        logger.info('batch_size: {}'.format(FLAGS.batch_size))
        logger.info('training_data_step: {}'.format(FLAGS.training_data_step))
        logger.info('model_size: {}'.format(FLAGS.model_size))
        logger.info('dataset: {}'.format(FLAGS.dataset))
        logger.info('stage: {}'.format(FLAGS.stage))
        logger.info('s1_steps: {}'.format(FLAGS.s1_steps))
        logger.info('logging_interval: {}'.format(FLAGS.logging_interval))
        logger.info('checkpoint_interval: {}'.format(FLAGS.checkpoint_interval))
        logger.info('save_summary: {}'.format(FLAGS.save_summary))

        assert FLAGS.stage in ['s1', 's2', 's1s2'], '{} is not valid.'.format(FLAGS.stage)
        assert FLAGS.subset in ['train', 'test'], '{} is not valid.'.format(FLAGS.subset)
        assert FLAGS.dataset in ['ucf101', 'ucf101_256', 'middlebury'], '{} is not valid.'.format(FLAGS.dataset)

        logger.info('Output_directory: {}'.format(out_dir))

        if FLAGS.model_size == 'large':
            from CyclicGen_model_large import Voxel_flow_model
        else:
            from CyclicGen_model import Voxel_flow_model

        if FLAGS.subset == 'train':
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

            if FLAGS.dataset == 'ucf101':
                data_list_path_frame1 = "data_list/ucf101_train_files_frame1.txt"
                data_list_path_frame2 = "data_list/ucf101_train_files_frame2.txt"
                data_list_path_frame3 = "data_list/ucf101_train_files_frame3.txt"
            if FLAGS.dataset == 'ucf101_256':
                data_list_path_frame1 = "data_list/ucf101_256_train_files_frame1.txt"
                data_list_path_frame2 = "data_list/ucf101_256_train_files_frame2.txt"
                data_list_path_frame3 = "data_list/ucf101_256_train_files_frame3.txt"
            if FLAGS.dataset == 'middlebury':
                data_list_path_frame1 = "data_list/middlebury_train_files_frame1.txt"
                data_list_path_frame2 = "data_list/middlebury_train_files_frame2.txt"
                data_list_path_frame3 = "data_list/middlebury_train_files_frame3.txt"

            ucf101_dataset_frame1 = dataset.Dataset(data_list_path_frame1)
            ucf101_dataset_frame2 = dataset.Dataset(data_list_path_frame2)
            ucf101_dataset_frame3 = dataset.Dataset(data_list_path_frame3)

            train(ucf101_dataset_frame1, ucf101_dataset_frame2, ucf101_dataset_frame3, out_dir, log_sep=' ,', hist_logger=hist_logger)


        elif FLAGS.subset == 'test':
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

            if FLAGS.dataset == 'ucf101':
                data_list_path_frame1 = "data_list/ucf101_test_files_frame1.txt"
                data_list_path_frame2 = "data_list/ucf101_test_files_frame2.txt"
                data_list_path_frame3 = "data_list/ucf101_test_files_frame3.txt"
            if FLAGS.dataset == 'ucf101_256':
                data_list_path_frame1 = "data_list/ucf101_256_train_files_frame1.txt"
                data_list_path_frame2 = "data_list/ucf101_256_train_files_frame2.txt"
                data_list_path_frame3 = "data_list/ucf101_256_train_files_frame3.txt"
            if FLAGS.dataset == 'middlebury':
                data_list_path_frame1 = "data_list/middlebury_test_files_frame1.txt"
                data_list_path_frame2 = "data_list/middlebury_test_files_frame2.txt"
                data_list_path_frame3 = "data_list/middlebury_test_files_frame3.txt"

            ucf101_dataset_frame1 = dataset.Dataset(data_list_path_frame1)
            ucf101_dataset_frame2 = dataset.Dataset(data_list_path_frame2)
            ucf101_dataset_frame3 = dataset.Dataset(data_list_path_frame3)

            test(ucf101_dataset_frame1, ucf101_dataset_frame2, ucf101_dataset_frame3)

    except:
        logger.exception('### An Exception occured')