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

FLAGS = tf.app.flags.FLAGS

# Define necessary FLAGS
tf.app.flags.DEFINE_string('train_dir', './train_dir',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_image_dir', './voxel_flow_train_image',
                           """Directory where to output images.""")
tf.app.flags.DEFINE_string('test_image_dir', './voxel_flow_test_image_baseline',
                           """Directory where to output images.""")
tf.app.flags.DEFINE_string('subset', 'train',
                           """Either 'train' or 'validation'.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', None,
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")
tf.app.flags.DEFINE_integer('max_steps', 10000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer(
    'batch_size', 8, 'The number of samples in each batch.')
tf.app.flags.DEFINE_float('initial_learning_rate', 0.00001,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_integer('training_data_step', 1, """The step used to reduce training data size""")
tf.app.flags.DEFINE_string('model_size', 'large', """The size of model""") ##
tf.app.flags.DEFINE_string('dataset', None, """The size of model ucf101_256 or middlebury """) ##
tf.app.flags.DEFINE_string('stage', 's1s2', """stage (s1 or s2)""") ##
tf.app.flags.DEFINE_integer('s1_steps', 20000, """ number of steps for stage1 if 's1s2' is specified as stage. """) ##

def _read_image(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string, channels=3)
    # image_decoded.set_shape([256, 256, 3])
    return tf.cast(image_decoded, dtype=tf.float32) / 127.5 - 1.0

def random_scaling(image, seed=1):
    scaling = tf.random_uniform([], 0.4, 0.6, seed=seed)
    return tf.image.resize_images(image, [tf.cast(tf.round(256*scaling), tf.int32), tf.cast(tf.round(256*scaling), tf.int32)])


def train(dataset_frame1, dataset_frame2, dataset_frame3, csv_logger, out_dir):
    """Trains a model."""
    with tf.Graph().as_default():
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
        if FLAGS.stage == 's2':
            edge_vgg_2 = Vgg16(input2,reuse=True)
        edge_vgg_3 = Vgg16(input3,reuse=True)

        edge_1 = tf.nn.sigmoid(edge_vgg_1.fuse)
        if FLAGS.stage == 's2':
            edge_2 = tf.nn.sigmoid(edge_vgg_2.fuse)
        edge_3 = tf.nn.sigmoid(edge_vgg_3.fuse)

        edge_1 = tf.reshape(edge_1,[-1,input1.get_shape().as_list()[1],input1.get_shape().as_list()[2],1])
        if FLAGS.stage == 's2':
            edge_2 = tf.reshape(edge_2,[-1,input1.get_shape().as_list()[1],input1.get_shape().as_list()[2],1])
        edge_3 = tf.reshape(edge_3,[-1,input1.get_shape().as_list()[1],input1.get_shape().as_list()[2],1])

        if FLAGS.stage == 's1':
            with tf.variable_scope("Cycle_DVF"):
                model1_s1_i00_i20 = Voxel_flow_model()
                prediction1, flow1 = model1_s1_i00_i20.inference(tf.concat([input1, input3, edge_1, edge_3], 3))
                reconstruction_loss = model1_s1_i00_i20.l1loss(prediction1, input2)

        if FLAGS.stage == 's2':
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
        print('all layers:')
        for var in t_vars: print(var.name)
        dof_vars = [var for var in t_vars if not 'hed' in var.name]
        print('optimize layers:')
        for var in dof_vars: print(var.name)

        if FLAGS.stage == 's1':
            cycle_consistency_loss = tf.convert_to_tensor(0.0, dtype=tf.float32)
            motion_linearity_loss =  tf.convert_to_tensor(0.0, dtype=tf.float32)
            total_loss = reconstruction_loss

        if FLAGS.stage == 's2':
            motion_linearity_loss = tf.reduce_mean(tf.square(model4_s2_i00_i20.flow - model3_s2_i05_i15.flow * 2.0))
            total_loss = reconstruction_loss + cycle_consistency_loss + 0.1*motion_linearity_loss

        # Perform learning rate scheduling.
        learning_rate = FLAGS.initial_learning_rate
        if learning_rate is None:
            if FLAGS.stage == 's1':
                learning_rate = 0.0001
            elif FLAGS.stage == 's2':
                learning_rate = 0.00001

        # Create an optimizer that performs gradient descent.
        with tf.variable_scope(tf.get_variable_scope(), reuse=None):
            update_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, var_list=dof_vars)

        # Create summaries
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries.append(tf.summary.scalar('total_loss', total_loss))
        summaries.append(tf.summary.image('input1', input1, 3))
        summaries.append(tf.summary.image('input2', input2, 3))
        summaries.append(tf.summary.image('input3', input3, 3))
        summaries.append(tf.summary.image('edge_1', edge_1, 3))
        if FLAGS.stage == 's2':
            summaries.append(tf.summary.image('edge_2', edge_1, 3))
        summaries.append(tf.summary.image('edge_3', edge_1, 3))

        if FLAGS.stage == 's1':
            summaries.append(tf.summary.image('prediction1', prediction1, 3))
        if FLAGS.stage == 's2':
            summaries.append(tf.summary.image('prediction3', prediction3, 3))
            summaries.append(tf.summary.image('prediction4', prediction4, 3))

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=50)

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge_all()

        # Restore checkpoint from file.
        last_step = -1
        if FLAGS.pretrained_model_checkpoint_path:
            sess = tf.Session()
            restorer = tf.train.Saver()
            restorer.restore(sess, FLAGS.pretrained_model_checkpoint_path)
            print('%s: Pre-trained model restored from %s' %
                  (datetime.now(), FLAGS.pretrained_model_checkpoint_path))
            try:
                last_step = int(str(FLAGS.pretrained_model_checkpoint_path).split(sep='-')[-1])
            except ValueError:
                print('The step number could not retrieved from the checkpoint path.'
                      'Continue running.')
            sess.run([batch_frame1.initializer, batch_frame2.initializer, batch_frame3.initializer])
        else:
            # Build an initialization operation to run below.
            init = tf.initialize_all_variables()
            sess = tf.Session()
            sess.run([init, batch_frame1.initializer, batch_frame2.initializer, batch_frame3.initializer])

        meta_model_file = 'hed_model/new-model.ckpt'
        saver2 = tf.train.Saver(var_list=[v for v in tf.all_variables() if "hed" in v.name])
        saver2.restore(sess, meta_model_file)

        # Summary Writter
        summary_writer = tf.summary.FileWriter(
            out_dir,
            graph=sess.graph)

        data_size = len(data_list_frame1)
        num_batches_per_epoch = int(data_size / FLAGS.batch_size)

        for step in range(last_step+1, FLAGS.max_steps):
            batch_idx = step % num_batches_per_epoch

            # Run single step update.
            _, total_loss_, \
                reconstruction_loss_, cycle_consistency_loss_, motion_linearity_loss_ = \
                sess.run([update_op, total_loss,
                          reconstruction_loss, cycle_consistency_loss, motion_linearity_loss])

            if batch_idx == 0:
                print('Epoch Number: %d' % int(step // num_batches_per_epoch))

            if step % 10 == 0:
                print('Step: {} | Loss: {} = {} + {} + {}'.format(step,
                                                                   total_loss_,
                                                                   reconstruction_loss_,
                                                                   cycle_consistency_loss_,
                                                                   motion_linearity_loss_))
                csv_logger(step,
                           total_loss_,
                           reconstruction_loss_,
                           cycle_consistency_loss_,
                           motion_linearity_loss_)

            # Save checkpoint
            if step % 2000 == 0 or (step + 1) == FLAGS.max_steps:
                # Output Summary
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
                #
                checkpoint_path = os.path.join(out_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


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
            print('%s: Pre-trained model restored from %s' %
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

            print(np.sum(batch_data_mask))
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
        print("Overall PSNR: %f db" % (PSNR / i))
        print("Overall SSIM: %f db" % (SSIM / i))


try:
    from table_logger import TableLogger
    LOGGING = True
except:
    print('Continue running without logging to a CSV file as table-logger is not installed.'
          ' To enable logging, "pip install table-logger" and rerun this code.')
    LOGGING = False


def timestamp():
    return datetime.now().strftime('%Y-%m-%dT%H%M%S')


if __name__ == '__main__':
    assert FLAGS.stage in ['s1', 's2']
    assert FLAGS.subset in ['train', 'test']
    assert FLAGS.dataset in ['ucf101', 'ucf101_256', 'middlebury']
    start_time = timestamp()
    config_str = '_'.join([FLAGS.model_size, FLAGS.dataset, FLAGS.stage, start_time])
    out_dir = FLAGS.train_dir + '/' + config_str
    Path(out_dir).mkdir(parents=True)
    print('Output_directory: ', out_dir)

    csv_logger = None
    if LOGGING:
        file_name = out_dir + '/log_{}.csv'.format(config_str)
        csv_logger = TableLogger(csv=True, file=file_name,
                         columns='Step,Loss,Reconstruction_Loss,Cycle_Consistency_Loss,Motion_Linearity_Loss',
                         rownum=True, time_delta=True, timestamp=True,
                         float_format='{:f}'.format)
        print('The loss values will be logged to: {}'.format(file_name))
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

        train(ucf101_dataset_frame1, ucf101_dataset_frame2, ucf101_dataset_frame3, csv_logger, out_dir)

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