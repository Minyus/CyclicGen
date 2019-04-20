"""CyclicGen_main.py"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import numpy as np
# import os
# import tensorflow as tf
from datetime import datetime, timedelta

from skimage.measure import compare_ssim as ssim
from pathlib import Path
from logging import getLogger
import logging


# from cyclicgen import dataset #import dataset
""" dataset.py  """

"""Implements a dataset class for handling image data"""

# from utils.image_utils import imread, imsave

DATA_PATH_BASE = '/home/VoxelFlow/dataset/ucf101_triplets/'


class Dataset(object):
    def __init__(self, data_list_file=None):
        """
          Args:
        """
        self.data_list_file = data_list_file

    def read_data_list_file(self):
        """Reads the data list_file into python list
        """
        f = open(self.data_list_file)
        data_list = [line.rstrip() for line in f]
        self.data_list = data_list
        return data_list

    def process_func(self, example_line):
        return imread(example_line)

"""   """


def tf_imread(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string, channels=3)
    # image_decoded.set_shape([256, 256, 3])
    return tf.cast(image_decoded, dtype=tf.float32) / 127.5 - 1.0 # value between -1 and +1


# from cyclicgen.utils import imwrite # from utils.image_utils import imwrite
""" image_utils.py  """
import scipy as sp
from scipy import misc


def imread(filename):
  """Read image from file.
  Args:
    filename: .
  Returns:
    im_array: .
  """
  im = sp.misc.imread(filename)
  # return im / 255.0
  return im / 127.5 - 1.0


def imwrite(filename, np_image):
  """Save image to file.
  Args:
    filename: .
    np_image: .
  """
  np_image = np.clip(np_image, -1.0, 1.0)
  # im = sp.misc.toimage(np_image, cmin=0, cmax=1.0)
  im = sp.misc.toimage(np_image, cmin=-1.0, cmax=1.0)
  im.save(filename)
"""   """





# from cyclicgen.vgg16 import Vgg16 # vgg16 import Vgg16
""" vgg16.py  """
# Adapted from : VGG 16 model : https://github.com/machrisaa/tensorflow-vgg
import time
import os
import inspect

import numpy as np
from termcolor import colored
import tensorflow as tf

# from losses import sigmoid_cross_entropy_balanced
""" losses.py  """
import tensorflow as tf


def sigmoid_cross_entropy_balanced(logits, label, name='cross_entropy_loss'):
    """
    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to
    tf.nn.weighted_cross_entropy_with_logits
    """
    y = tf.cast(label, tf.float32)

    count_neg = tf.reduce_sum(1. - y)
    count_pos = tf.reduce_sum(y)

    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = tf.reduce_mean(cost * (1 - beta))

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost, name=name)

"""   """


# import pdb
#from io import IO

VGG_MEAN = [103.939, 116.779, 123.68]
class Vgg16():

    def __init__(self, input_image,reuse=None):

        # self.cfgs 1= cfgs
#        self.io = IO()

        base_path = os.path.abspath(os.path.dirname(__file__))
        if __name__ == '__main__':
            base_path = os.getcwd()
        weights_file = os.path.join(base_path, 'vgg16.npy')

        self.data_dict = np.load(weights_file, encoding='latin1').item()
        # self.io.print_info("Model weights loaded from {}".format(self.cfgs['model_weights_path']))

        rgb_scaled = tf.subtract((input_image+tf.ones_like(input_image)),2)*255.
        red, green, blue = tf.split(rgb_scaled, 3, 3)

        self.images = tf.concat([blue - VGG_MEAN[0],
                        green - VGG_MEAN[1],
                        red - VGG_MEAN[2]],
                        3)
        # self.images = tf.placeholder(tf.float32, [None, self.cfgs[run]['image_height'], self.cfgs[run]['image_width'], self.cfgs[run]['n_channels']])
        # self.edgemaps = tf.placeholder(tf.float32, [None, self.cfgs[run]['image_height'], self.cfgs[run]['image_width'], 1])
        self.define_model(reuse=reuse)

    def define_model(self,reuse=None):

        """
        Load VGG params from disk without FC layers A
        Add branch layers (with deconv) after each CONV block
        """
        with tf.variable_scope('hed'):
            start_time = time.time()
            self.conv1_1 = self.conv_layer_vgg(self.images, "conv1_1")
            self.conv1_2 = self.conv_layer_vgg(self.conv1_1, "conv1_2")
            self.side_1 = self.side_layer(self.conv1_2, "side_1", 1,reuse=reuse)
            self.pool1 = self.max_pool(self.conv1_2, 'pool1')

            # self.io.print_info('Added CONV-BLOCK-1+SIDE-1')

            self.conv2_1 = self.conv_layer_vgg(self.pool1, "conv2_1")
            self.conv2_2 = self.conv_layer_vgg(self.conv2_1, "conv2_2")
            self.side_2 = self.side_layer(self.conv2_2, "side_2", 2,reuse=reuse)
            self.pool2 = self.max_pool(self.conv2_2, 'pool2')

            # self.io.print_info('Added CONV-BLOCK-2+SIDE-2')

            self.conv3_1 = self.conv_layer_vgg(self.pool2, "conv3_1")
            self.conv3_2 = self.conv_layer_vgg(self.conv3_1, "conv3_2")
            self.conv3_3 = self.conv_layer_vgg(self.conv3_2, "conv3_3")
            self.side_3 = self.side_layer(self.conv3_3, "side_3", 4,reuse=reuse)
            self.pool3 = self.max_pool(self.conv3_3, 'pool3')

            # self.io.print_info('Added CONV-BLOCK-3+SIDE-3')

            self.conv4_1 = self.conv_layer_vgg(self.pool3, "conv4_1")
            self.conv4_2 = self.conv_layer_vgg(self.conv4_1, "conv4_2")
            self.conv4_3 = self.conv_layer_vgg(self.conv4_2, "conv4_3")
            self.side_4 = self.side_layer(self.conv4_3, "side_4", 8,reuse=reuse)
            self.pool4 = self.max_pool(self.conv4_3, 'pool4')

            # self.io.print_info('Added CONV-BLOCK-4+SIDE-4')

            self.conv5_1 = self.conv_layer_vgg(self.pool4, "conv5_1")
            self.conv5_2 = self.conv_layer_vgg(self.conv5_1, "conv5_2")
            self.conv5_3 = self.conv_layer_vgg(self.conv5_2, "conv5_3")
            self.side_5 = self.side_layer(self.conv5_3, "side_5", 16,reuse=reuse)

            # self.io.print_info('Added CONV-BLOCK-5+SIDE-5')

            self.side_outputs = [self.side_1, self.side_2, self.side_3, self.side_4, self.side_5]
            w_shape = [1, 1, len(self.side_outputs), 1]
            if reuse == True:
                tf.get_variable_scope().reuse_variables()
            self.fuse = self.conv_layer(tf.concat(self.side_outputs, axis=3),
                                    w_shape, name='fuse_1', use_bias=False,
                                    w_init=tf.constant_initializer(0.2))
            #tf.get_variable_scope().reuse == False

            # self.io.print_info('Added FUSE layer')

            # complete output maps from side layer and fuse layers
            self.outputs = self.side_outputs + [self.fuse]

            self.data_dict = None
            # self.io.print_info("Build model finished: {:.4f}s".format(time.time() - start_time))

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer_vgg(self, bottom, name):
        """
            Adding a conv layer + weight parameters from a dict
        """
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def conv_layer(self, x, W_shape, b_shape=None, name=None,
                   padding='SAME', use_bias=True, w_init=None, b_init=None):

        W = self.weight_variable(W_shape, w_init, 'Variable')
        tf.summary.histogram('weights_{}'.format(name), W)

        if use_bias:
            b = self.bias_variable([b_shape], b_init, 'Variable_1')
            tf.summary.histogram('biases_{}'.format(name), b)

        conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

        return conv + b if use_bias else conv

    def deconv_layer(self, x, upscale, name, padding='SAME', w_init=None):

        x_shape = tf.shape(x)
        in_shape = x.shape.as_list()

        w_shape = [upscale * 2, upscale * 2, in_shape[-1], 1]
        strides = [1, upscale, upscale, 1]

        W = self.weight_variable(w_shape, w_init, 'Variable_2')
        tf.summary.histogram('weights_{}'.format(name), W)

        out_shape = tf.stack([x_shape[0], x_shape[1], x_shape[2], w_shape[2]]) * tf.constant(strides, tf.int32)
        deconv = tf.nn.conv2d_transpose(x, W, out_shape, strides=strides, padding=padding)

        return deconv

    def side_layer(self, inputs, name, upscale,reuse=None):
        """
            https://github.com/s9xie/hed/blob/9e74dd710773d8d8a469ad905c76f4a7fa08f945/examples/hed/train_val.prototxt#L122
            1x1 conv followed with Deconvoltion layer to upscale the size of input image sans color
        """
        with tf.variable_scope(name,reuse=reuse):

            in_shape = inputs.shape.as_list()
            w_shape = [1, 1, in_shape[-1], 1]

            classifier = self.conv_layer(inputs, w_shape, b_shape=1,
                                         w_init=tf.constant_initializer(),
                                         b_init=tf.constant_initializer(),
                                         name=name + '_reduction')

            classifier = self.deconv_layer(classifier, upscale=upscale,
                                           name='{}_deconv_{}'.format(name, upscale),
                                           w_init=tf.truncated_normal_initializer(stddev=0.1))

            return classifier

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def weight_variable(self, shape, initial, name):

        return tf.get_variable(name, shape=shape, initializer=initial)

    def bias_variable(self, shape, initial, name):

        return tf.get_variable(name, shape=shape, initializer=initial)

    def setup_testing(self, session):

        """
            Apply sigmoid non-linearity to side layer ouputs + fuse layer outputs for predictions
        """

        self.predictions = []

        for idx, b in enumerate(self.outputs):
            output = tf.nn.sigmoid(b, name='output_{}'.format(idx))
            self.predictions.append(output)

    def setup_training(self, session):

        """
            Apply sigmoid non-linearity to side layer ouputs + fuse layer outputs
            Compute total loss := side_layer_loss + fuse_layer_loss
            Compute predicted edge maps from fuse layer as pseudo performance metric to track
        """

        self.predictions = []
        self.loss = 0

        self.io.print_warning('Deep supervision application set to {}'.format(self.cfgs['deep_supervision']))

        for idx, b in enumerate(self.side_outputs):
            output = tf.nn.sigmoid(b, name='output_{}'.format(idx))
            cost = sigmoid_cross_entropy_balanced(b, self.edgemaps, name='cross_entropy{}'.format(idx))

            self.predictions.append(output)
            if self.cfgs['deep_supervision']:
                self.loss += (self.cfgs['loss_weights'] * cost)

        fuse_output = tf.nn.sigmoid(self.fuse, name='fuse')
        fuse_cost = sigmoid_cross_entropy_balanced(self.fuse, self.edgemaps, name='cross_entropy_fuse')

        self.predictions.append(fuse_output)
        self.loss += (self.cfgs['loss_weights'] * fuse_cost)

        pred = tf.cast(tf.greater(fuse_output, 0.5), tf.int32, name='predictions')
        error = tf.cast(tf.not_equal(pred, tf.cast(self.edgemaps, tf.int32)), tf.float32)
        self.error = tf.reduce_mean(error, name='pixel_error')

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('error', self.error)

        self.merged_summary = tf.summary.merge_all()

        self.train_writer = tf.summary.FileWriter(self.cfgs['save_dir'] + '/train', session.graph)
        self.val_writer = tf.summary.FileWriter(self.cfgs['save_dir'] + '/val')

"""   """

# from CyclicGen_model_large import Voxel_flow_model
""" CyclicGen_model_large.py  """
"""Implements a voxel flow model."""

import tensorflow as tf
import tensorflow.contrib.slim as slim
# from utils.loss_utils import l1_loss #, l2_loss, vae_loss
""" loss_utils.py  """

def l1_loss(predictions, targets):
  """Implements tensorflow l1 loss.
  Args:
  Returns:
  """
  total_elements = (tf.shape(targets)[0] * tf.shape(targets)[1] * tf.shape(targets)[2]
      * tf.shape(targets)[3])
  total_elements = tf.to_float(total_elements)

  loss = tf.reduce_sum(tf.abs(predictions- targets))
  loss = tf.div(loss, total_elements)
  return loss
"""   """


# from utils.geo_layer_utils import bilinear_interp
# from utils.geo_layer_utils import meshgrid
""" geo_layer_utils.py """


def bilinear_interp(im, x, y, name):
    """Perform bilinear sampling on im given x, y coordinates

    This function implements the differentiable sampling mechanism with
    bilinear kernel. Introduced in https://arxiv.org/abs/1506.02025, equation
    (5).

    x,y are tensors specfying normalized coorindates [-1,1] to sample from im.
    (-1,1) means (0,0) coordinate in im. (1,1) means the most bottom right pixel.

    Args:
      im: Tensor of size [batch_size, height, width, depth]
      x: Tensor of size [batch_size, height, width, 1]
      y: Tensor of size [batch_size, height, width, 1]
      name: String for the name for this opt.
    Returns:
      Tensor of size [batch_size, height, width, depth]
    """
    with tf.variable_scope(name):
        x = tf.reshape(x, [-1])
        y = tf.reshape(y, [-1])

        # constants
        num_batch = tf.shape(im)[0]
        _, height, width, channels = im.get_shape().as_list()

        x = tf.to_float(x)
        y = tf.to_float(y)

        height_f = tf.cast(height, 'float32')
        width_f = tf.cast(width, 'float32')
        zero = tf.constant(0, dtype=tf.int32)

        max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')
        max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
        x = (x + 1.0) * (width_f - 1.0) / 2.0
        y = (y + 1.0) * (height_f - 1.0) / 2.0

        # Sampling
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        dim2 = width
        dim1 = width * height

        # Create base index
        base = tf.range(num_batch) * dim1
        base = tf.reshape(base, [-1, 1])
        base = tf.tile(base, [1, height * width])
        base = tf.reshape(base, [-1])

        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # Use indices to look up pixels
        im_flat = tf.reshape(im, tf.stack([-1, channels]))
        im_flat = tf.to_float(im_flat)
        pixel_a = tf.gather(im_flat, idx_a)
        pixel_b = tf.gather(im_flat, idx_b)
        pixel_c = tf.gather(im_flat, idx_c)
        pixel_d = tf.gather(im_flat, idx_d)

        # Interpolate the values
        x1_f = tf.to_float(x1)
        y1_f = tf.to_float(y1)

        wa = tf.expand_dims((x1_f - x) * (y1_f - y), 1)
        wb = tf.expand_dims((x1_f - x) * (1.0 - (y1_f - y)), 1)
        wc = tf.expand_dims((1.0 - (x1_f - x)) * (y1_f - y), 1)
        wd = tf.expand_dims((1.0 - (x1_f - x)) * (1.0 - (y1_f - y)), 1)

        output = tf.add_n([wa * pixel_a, wb * pixel_b, wc * pixel_c, wd * pixel_d])
        output = tf.reshape(output, shape=tf.stack([num_batch, height, width, channels]))
        return output


def meshgrid(height, width):
    """Tensorflow meshgrid function.
    """
    with tf.variable_scope('meshgrid'):
        x_t = tf.matmul(
            tf.ones(shape=tf.stack([height, 1])),
            tf.transpose(
                tf.expand_dims(
                    tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
        y_t = tf.matmul(
            tf.expand_dims(
                tf.linspace(-1.0, 1.0, height), 1),
            tf.ones(shape=tf.stack([1, width])))
        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))
        # grid_x = tf.reshape(x_t_flat, [1, height, width, 1])
        # grid_y = tf.reshape(y_t_flat, [1, height, width, 1])
        grid_x = tf.reshape(x_t_flat, [1, height, width])
        grid_y = tf.reshape(y_t_flat, [1, height, width])
        return grid_x, grid_y


"""   """



FLAGS = tf.app.flags.FLAGS
epsilon = 0.001


class VoxelFlowModel(object):
    def __init__(self, batch_size, is_train=True, adaptive_temporal_flow=False):
        self.is_train = is_train
        self.adaptive_temporal_flow = adaptive_temporal_flow
        self.batch_size = batch_size

    def inference(self, input_images, target_time_point=0.5):
        """Inference on a set of input_images.
        Args:
        """
        return self._build_model(input_images, target_time_point=target_time_point)

    def total_var(self, images):
        pixel_dif1 = images[:, 1:, :, :] - images[:, :-1, :, :]
        pixel_dif2 = images[:, :, 1:, :] - images[:, :, :-1, :]
        tot_var = (tf.reduce_mean(tf.sqrt(tf.square(pixel_dif1) + epsilon**2)) + tf.reduce_mean(tf.sqrt(tf.square(pixel_dif2) + epsilon**2)))
        return tot_var

    def loss(self, predictions, targets):
        """Compute the necessary loss for training.
        Args:
        Returns:
        """
        # self.reproduction_loss = l1_loss(predictions, targets)
        self.reproduction_loss = tf.reduce_mean(tf.sqrt(tf.square(predictions - targets) + epsilon**2))

        self.motion_loss = self.total_var(self.flow)
        self.mask_loss = self.total_var(self.mask)

        # return [self.reproduction_loss, self.prior_loss]
        return self.reproduction_loss + 0.01 * self.motion_loss + 0.005 * self.mask_loss

    def l1loss(self, predictions, targets):
        self.reproduction_loss = l1_loss(predictions, targets)
        return self.reproduction_loss

    def _build_model(self, input_images, target_time_point=0.5):
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.leaky_relu,
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            weights_regularizer=slim.l2_regularizer(0.0001)):
            # Define network
            batch_norm_params = {
                'decay': 0.9997,
                'epsilon': 0.001,
                'is_training': self.is_train,
            }
            with slim.arg_scope([slim.batch_norm], is_training=self.is_train, updates_collections=None):
                with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                                    normalizer_params=batch_norm_params):
                    x0 = slim.conv2d(input_images, 32, [7, 7], stride=1, scope='conv1')
                    x0_1 = slim.conv2d(x0, 32, [7, 7], stride=1, scope='conv1_1')

                    net = slim.avg_pool2d(x0_1, [2, 2], scope='pool1')
                    x1 = slim.conv2d(net, 64, [5, 5], stride=1, scope='conv2')
                    x1_1 = slim.conv2d(x1, 64, [5, 5], stride=1, scope='conv2_1')

                    net = slim.avg_pool2d(x1_1, [2, 2], scope='pool2')
                    x2 = slim.conv2d(net, 128, [3, 3], stride=1, scope='conv3')
                    x2_1 = slim.conv2d(x2, 128, [3, 3], stride=1, scope='conv3_1')

                    net = slim.avg_pool2d(x2_1, [2, 2], scope='pool3')
                    x3 = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv4')
                    x3_1 = slim.conv2d(x3, 256, [3, 3], stride=1, scope='conv4_1')

                    net = slim.avg_pool2d(x3_1, [2, 2], scope='pool4')
                    x4 = slim.conv2d(net, 512, [3, 3], stride=1, scope='conv5')
                    x4_1 = slim.conv2d(x4, 512, [3, 3], stride=1, scope='conv5_1')

                    net = slim.avg_pool2d(x4_1, [2, 2], scope='pool5')
                    net = slim.conv2d(net, 512, [3, 3], stride=1, scope='conv6')
                    net = slim.conv2d(net, 512, [3, 3], stride=1, scope='conv6_1')

                    net = tf.image.resize_bilinear(net, [x4.get_shape().as_list()[1], x4.get_shape().as_list()[2]])
                    net = slim.conv2d(tf.concat([net, x4_1], -1), 512, [3, 3], stride=1, scope='conv7')
                    net = slim.conv2d(net, 512, [3, 3], stride=1, scope='conv7_1')

                    net = tf.image.resize_bilinear(net, [x3.get_shape().as_list()[1], x3.get_shape().as_list()[2]])
                    net = slim.conv2d(tf.concat([net, x3_1], -1), 256, [3, 3], stride=1, scope='conv8')
                    net = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv8_1')

                    net = tf.image.resize_bilinear(net, [x2.get_shape().as_list()[1], x2.get_shape().as_list()[2]])
                    net = slim.conv2d(tf.concat([net, x2_1], -1), 128, [3, 3], stride=1, scope='conv9')
                    net = slim.conv2d(net, 128, [3, 3], stride=1, scope='conv9_1')

                    net = tf.image.resize_bilinear(net, [x1.get_shape().as_list()[1], x1.get_shape().as_list()[2]])
                    net = slim.conv2d(tf.concat([net, x1_1], -1), 64, [3, 3], stride=1, scope='conv10')
                    net = slim.conv2d(net, 64, [3, 3], stride=1, scope='conv10_1')

                    net = tf.image.resize_bilinear(net, [x0.get_shape().as_list()[1], x0.get_shape().as_list()[2]])
                    net = slim.conv2d(tf.concat([net, x0_1], -1), 32, [3, 3], stride=1, scope='conv11')
                    y0 = slim.conv2d(net, 32, [3, 3], stride=1, scope='conv11_1')

        net = slim.conv2d(y0, 3, [5, 5], stride=1, activation_fn=tf.tanh,
                          normalizer_fn=None, scope='conv12')
        net_copy = net

        flow = net[:, :, :, 0:2] #_ (B,H,W,2)
        self.flow = flow #_ (B,H,W,2)

        temporal_flow_for_pixel = net[:, :, :, 2] #_ (B,H,W)
        mask_for_pixel = 0.5 * (1.0 + temporal_flow_for_pixel) #_ (B,H,W)

        target_time_point_for_pixel = target_time_point * tf.ones_like(mask_for_pixel) #_ (B,H,W)
        if self.adaptive_temporal_flow:
            target_time_point_for_pixel = 1.0-mask_for_pixel #_ (B,H,W)

        grid_x, grid_y = meshgrid(x0.get_shape().as_list()[1], x0.get_shape().as_list()[2])
        grid_x = tf.tile(grid_x, [self.batch_size, 1, 1])
        grid_y = tf.tile(grid_y, [self.batch_size, 1, 1])

        flow_ratio = tf.constant([255.0 / (x0.get_shape().as_list()[2]-1), 255.0 / (x0.get_shape().as_list()[1]-1)]) #_ (2,)
        flow = flow * tf.expand_dims(tf.expand_dims(tf.expand_dims(flow_ratio, 0), 0), 0) #_ (1,1,1,2)

        coor_x_1 = grid_x + target_time_point_for_pixel * flow[:, :, :, 0] #_ (B,H,W)
        coor_y_1 = grid_y + target_time_point_for_pixel * flow[:, :, :, 1] #_ (B,H,W)

        coor_x_2 = grid_x + (target_time_point_for_pixel-1.0) * flow[:, :, :, 0] #_ (B,H,W)
        coor_y_2 = grid_y + (target_time_point_for_pixel-1.0) * flow[:, :, :, 1] #_ (B,H,W)

        output_1 = bilinear_interp(input_images[:, :, :, 0:3], coor_x_1, coor_y_1, 'interpolate') #_ (B,H,W,3)
        output_2 = bilinear_interp(input_images[:, :, :, 3:6], coor_x_2, coor_y_2, 'interpolate') #_ (B,H,W,3)

        self.warped_img1 = output_1 #_ (B,H,W,3)
        self.warped_img2 = output_2 #_ (B,H,W,3)

        self.warped_flow1 = bilinear_interp(-flow[:, :, :, 0:3]*0.5*0.5, coor_x_1, coor_y_1, 'interpolate') #_ (B,H,W,3)
        self.warped_flow2 = bilinear_interp(flow[:, :, :, 0:3]*0.5*0.5, coor_x_2, coor_y_2, 'interpolate') #_ (B,H,W,3)

        mask = tf.expand_dims(mask_for_pixel, 3) #_ (B,H,W,1)
        mask = tf.clip_by_value(mask, 0.0, 1.0)
        self.mask = mask #_ (B,H,W,1)
        mask_rgb = tf.tile(mask, [1, 1, 1, 3]) #_ (B,H,W,3)
        net = tf.multiply(mask_rgb, output_1) + tf.multiply(1.0 - mask_rgb, output_2) #_ (B,H,W,3)

        return [net, net_copy]

"""   """

""" main from here """
logger = getLogger(__name__)

FLAGS = tf.app.flags.FLAGS

# Define necessary FLAGS
tf.app.flags.DEFINE_string('train_dir', './train_dir',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_string('task', 'train_test_generate',
                           """ One or more of 'train', 'test', 'generate'.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', None,
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")
tf.app.flags.DEFINE_integer('max_steps', None,
                            """Number of steps to run. if None, steps equivalent to max_epochs. """)
tf.app.flags.DEFINE_integer('max_epochs', 10,
                            """Number of epochs to run. """)
tf.app.flags.DEFINE_integer('batch_size', 8, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer('training_data_step', 1, """The step used to reduce training data size""")
tf.app.flags.DEFINE_string('model_size', 'large', """The size of model""") ##
tf.app.flags.DEFINE_string('dataset_train', 'ucf101_256', """dataset_train (ucf101_256 or middlebury) """) ##
tf.app.flags.DEFINE_string('dataset_test', 'ucf101', """dataset_test (ucf101, ucf101_256, or middlebury) """) ##
tf.app.flags.DEFINE_string('stage', 's1s2', """stage (s1 or s2)""") ##
tf.app.flags.DEFINE_integer('s1_steps', None, """ number of steps for stage1 if 's1s2' is specified as stage. s1_epochs if None (defalt) """) ##
tf.app.flags.DEFINE_integer('s1_epochs', 5, """ number of epochs for stage1 if 's1s2' is specified as stage and s1_steps = None """) ##
tf.app.flags.DEFINE_integer('logging_interval', 10, """ number of steps of interval to log. """) ##
tf.app.flags.DEFINE_integer('checkpoint_interval', None, """ number of steps of interval to save checkpoints. if None, 1 epoch. """) ##
tf.app.flags.DEFINE_bool('save_summary', False, """ save summary if True.  """) ##
tf.app.flags.DEFINE_integer('graph_level_seed', 0, """ TensorFlow's Graph-level random seed. """) ##
tf.app.flags.DEFINE_integer('crop_size', 128, """ Crop size (width and height) """) ##
tf.app.flags.DEFINE_float('coef_loss_c', 1.0, """ coef_cycle_consistency_loss """) ##
tf.app.flags.DEFINE_float('coef_loss_m', 0.1, """ coef_motion_linearity_loss """) ##
tf.app.flags.DEFINE_float('coef_loss_s', 0.0, """ coef_stable_motion_loss """) ##
tf.app.flags.DEFINE_float('coef_loss_t', 0.0, """ coef_temporal_regularization_loss """) ##
tf.app.flags.DEFINE_string('strategy', 'original_cycle_gen', """ strategy: original_cycle_gen, adaptive_temporal_flow, accel_adjust """) ##
tf.app.flags.DEFINE_string('logging_level', 'info', """ logging_level """) ##
tf.app.flags.DEFINE_string('gen_i0_path', './Middlebury/eval-color-allframes/eval-data/Backyard/frame07.png', """ gen_i0_path """) ##
tf.app.flags.DEFINE_string('gen_i1_path', './Middlebury/eval-color-allframes/eval-data/Backyard/frame08.png', """ gen_i1_path """) ##
tf.app.flags.DEFINE_string('gen_out_path', None, """ gen_out_path """) ##



"""
def random_scaling(image, seed=1):
    scaling = tf.random_uniform([], 0.4, 0.6, seed=seed)
    return tf.image.resize_images(image, [tf.cast(tf.round(256*scaling), tf.int32), tf.cast(tf.round(256*scaling), tf.int32)])
"""

def train(dataset_frame1, dataset_frame2, dataset_frame3, out_dir, log_sep=' ,',hist_logger=None, seed=1,
          trained_model_checkpoint_path=None):
    """Trains a model."""

    tf.set_random_seed(FLAGS.graph_level_seed)
    crop_size = FLAGS.crop_size
    batch_size = FLAGS.batch_size
    graph = tf.Graph()
    with graph.as_default():
        s2_flag_tensor = tf.placeholder(dtype="float", shape=None)

        # Create input.
        data_list_frame1 = dataset_frame1.read_data_list_file()
        data_list_frame1 = data_list_frame1[::FLAGS.training_data_step]
        dataset_frame1 = tf.data.Dataset.from_tensor_slices(tf.constant(data_list_frame1))
        dataset_frame1 = dataset_frame1.apply(
            tf.contrib.data.shuffle_and_repeat(buffer_size=1000000, count=None, seed=seed)).map(tf_imread).map(
            lambda image: tf.image.random_flip_left_right(image, seed=seed)).map(
            lambda image: tf.image.random_flip_up_down(image, seed=seed)).map(
            lambda image: tf.random_crop(image, [crop_size, crop_size, 3], seed=seed)).map(
            lambda image: tf.expand_dims(image, 0)).map(
            lambda image: tf.image.resize_bilinear(image, [256, 256])).map(
            lambda image: image[0,:,:,:])
        dataset_frame1 = dataset_frame1.prefetch(batch_size)

        data_list_frame2 = dataset_frame2.read_data_list_file()
        data_list_frame2 = data_list_frame2[::FLAGS.training_data_step]
        dataset_frame2 = tf.data.Dataset.from_tensor_slices(tf.constant(data_list_frame2))
        dataset_frame2 = dataset_frame2.apply(
            tf.contrib.data.shuffle_and_repeat(buffer_size=1000000, count=None, seed=seed)).map(tf_imread).map(
            lambda image: tf.image.random_flip_left_right(image, seed=seed)).map(
            lambda image: tf.image.random_flip_up_down(image, seed=seed)).map(
            lambda image: tf.random_crop(image, [crop_size, crop_size, 3], seed=seed)).map(
            lambda image: tf.expand_dims(image, 0)).map(
            lambda image: tf.image.resize_bilinear(image, [256, 256])).map(
            lambda image: image[0,:,:,:])
        dataset_frame2 = dataset_frame2.prefetch(batch_size)


        data_list_frame3 = dataset_frame3.read_data_list_file()
        data_list_frame3 = data_list_frame3[::FLAGS.training_data_step]
        dataset_frame3 = tf.data.Dataset.from_tensor_slices(tf.constant(data_list_frame3))
        dataset_frame3 = dataset_frame3.apply(
            tf.contrib.data.shuffle_and_repeat(buffer_size=1000000, count=None, seed=seed)).map(tf_imread).map(
            lambda image: tf.image.random_flip_left_right(image, seed=seed)).map(
            lambda image: tf.image.random_flip_up_down(image, seed=seed)).map(
            lambda image: tf.random_crop(image, [crop_size, crop_size, 3], seed=seed)).map(
            lambda image: tf.expand_dims(image, 0)).map(
            lambda image: tf.image.resize_bilinear(image, [256, 256])).map(
            lambda image: image[0,:,:,:])
        dataset_frame3 = dataset_frame3.prefetch(batch_size)

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

        # if stage == 's1':
        #     with tf.variable_scope("Cycle_DVF"):
        #         model1_s1_i00_i20 = VoxelFlowModel(adaptive_temporal_flow=FLAGS.strategy == 'adaptive_temporal_flow')
        #         prediction1, _ = model1_s1_i00_i20.inference(tf.concat([input1, input3, edge_1, edge_3], 3))
        #         loss_c = model1_s1_i00_i20.l1loss(prediction1, input2)

        if True: #if stage == 's2':
            input_placeholder1 = tf.concat([input1, input2], 3)
            input_placeholder2 = tf.concat([input2, input3], 3)

            input_placeholder1 = tf.concat([input_placeholder1, edge_1, edge_2], 3)
            input_placeholder2 = tf.concat([input_placeholder2, edge_2, edge_3], 3)

            with tf.variable_scope("Cycle_DVF"):
                model1_s2_i00_i10 = VoxelFlowModel(batch_size=batch_size)
                prediction1, _ = model1_s2_i00_i10.inference(input_placeholder1)

            with tf.variable_scope("Cycle_DVF", reuse=True):
                model2_s2_i10_i20 = VoxelFlowModel(batch_size=batch_size)
                prediction2, _ = model2_s2_i10_i20.inference(input_placeholder2)

            edge_vgg_prediction1 = Vgg16(prediction1,reuse=True)
            edge_vgg_prediction2 = Vgg16(prediction2,reuse=True)

            edge_prediction1 = tf.nn.sigmoid(edge_vgg_prediction1.fuse)
            edge_prediction2 = tf.nn.sigmoid(edge_vgg_prediction2.fuse)

            edge_prediction1 = tf.reshape(edge_prediction1,[-1,input1.get_shape().as_list()[1],input1.get_shape().as_list()[2],1])
            edge_prediction2 = tf.reshape(edge_prediction2,[-1,input1.get_shape().as_list()[1],input1.get_shape().as_list()[2],1])

            adaptive_temporal_flow = FLAGS.strategy == 'adaptive_temporal_flow'
            accel_adjust = FLAGS.strategy == 'accel_adjust'

            target_time_point = 0.5

            if accel_adjust:
                f1 = model1_s2_i00_i10.flow #_ (B,H,W,2)
                f2 = (model1_s2_i00_i10.flow + model2_s2_i10_i20.flow) #_ (B,H,W,2)

                f1f2 = tf.reduce_sum(tf.multiply(f1, f2), axis=3, keepdims=False) #_ (B,H,W)
                f2f2 = tf.reduce_sum(tf.multiply(f2, f2), axis=3, keepdims=False) #_ (B,H,W)
                epsilon_pixel_sq = 1.0
                target_time_point = (f1f2 + epsilon_pixel_sq) / (f2f2 + 2.0 * epsilon_pixel_sq) #_ (B,H,W)

            with tf.variable_scope("Cycle_DVF", reuse=True):
                model3_s2_i05_i15 = VoxelFlowModel(batch_size=batch_size, adaptive_temporal_flow=adaptive_temporal_flow)
                prediction3, _ = model3_s2_i05_i15.inference(tf.concat([prediction1, prediction2, edge_prediction1, edge_prediction2], 3),
                                                             target_time_point=target_time_point)
                loss_c = model3_s2_i05_i15.l1loss(prediction3, input2)

            with tf.variable_scope("Cycle_DVF", reuse=True):
                model4_s2_i00_i20 = VoxelFlowModel(batch_size=batch_size, adaptive_temporal_flow=adaptive_temporal_flow)
                prediction4, _ = model4_s2_i00_i20.inference(tf.concat([input1, input3,edge_1,edge_3], 3),
                                                             target_time_point=target_time_point)
                loss_r = model4_s2_i00_i20.l1loss(prediction4, input2)

        t_vars = tf.trainable_variables()

        logger.debug('all_layers:' + ' | '.join([var.name for var in t_vars]))
        dof_vars = [var for var in t_vars if not 'hed' in var.name]

        logger.debug('optimize layers:' + ' | '.join([var.name for var in dof_vars]))

        # if False: #if stage == 's1':
        #     loss_c = tf.convert_to_tensor(0.0, dtype=tf.float32)
        #     loss_m =  tf.convert_to_tensor(0.0, dtype=tf.float32)
        #     total_loss = loss_r

        if True: #if stage == 's2':
            loss_m = tf.reduce_mean(tf.square(model4_s2_i00_i20.flow - model3_s2_i05_i15.flow * 2.0))
            loss_s = tf.reduce_mean(tf.square((model2_s2_i10_i20.flow - model1_s2_i00_i10.flow) * 2.0))
            loss_t = tf.reduce_mean(tf.square((model4_s2_i00_i20.mask - 0.5) * 2.0))
            total_loss = loss_r \
                         + s2_flag_tensor * FLAGS.coef_loss_c * loss_c \
                         + s2_flag_tensor * FLAGS.coef_loss_m * loss_m \
                         + s2_flag_tensor * FLAGS.coef_loss_s * loss_s \
                         + s2_flag_tensor * FLAGS.coef_loss_t * loss_t

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
            if trained_model_checkpoint_path:
                restorer = tf.train.Saver()
                restorer.restore(sess, trained_model_checkpoint_path)
                logger.info('%s: Pre-trained model restored from %s' %
                      (timestamp(), trained_model_checkpoint_path))
                try:
                    last_step = int(str(trained_model_checkpoint_path).split(sep='-')[-1])
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
            logger.info('data_size: {}'.format(data_size))
            num_steps_per_epoch = int(data_size // batch_size)
            logger.info('num_steps_per_epoch: {}'.format(num_steps_per_epoch))

            max_steps = FLAGS.max_steps
            if max_steps is None:
                max_steps = FLAGS.max_epochs * num_steps_per_epoch
            logger.info('max_steps: {}'.format(max_steps))

            if FLAGS.stage == 's1s2':
                s1_steps = FLAGS.s1_steps
                if s1_steps is None:
                    s1_steps = FLAGS.s1_epochs * num_steps_per_epoch
            if FLAGS.stage == 's2':
                s1_steps = 0
            if FLAGS.stage == 's1':
                s1_steps = max_steps
            logger.info('s1_steps: {}'.format(s1_steps))

            checkpoint_interval = FLAGS.checkpoint_interval
            if checkpoint_interval is None:
                checkpoint_interval = num_steps_per_epoch

            initial_step = last_step + 1

            total_loss_ssum = 0
            loss_r_ssum = 0
            loss_c_ssum = 0
            loss_m_ssum = 0
            loss_s_ssum = 0
            loss_t_ssum = 0

            for step_i in range(initial_step, max_steps):
                batch_idx = step_i % num_steps_per_epoch

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
                    logger.info('Epoch Number: %d' % int(step_i // num_steps_per_epoch))

                if True: # if step_i % FLAGS.logging_interval == 0:
                    total_loss_bsum, loss_r_bsum, loss_c_bsum, loss_m_bsum, loss_s_bsum, loss_t_bsum = \
                        sess.run([total_loss, loss_r, loss_c, loss_m, loss_s, loss_t],
                                 feed_dict={s2_flag_tensor: s2_flag})

                    total_loss_ssum += total_loss_bsum
                    loss_r_ssum += loss_r_bsum
                    loss_c_ssum += loss_c_bsum
                    loss_m_ssum += loss_m_bsum
                    loss_s_ssum += loss_s_bsum
                    loss_t_ssum += loss_t_bsum

                if step_i % FLAGS.logging_interval == (FLAGS.logging_interval-1):
                    total_loss_mean = total_loss_ssum / (FLAGS.logging_interval * batch_size)
                    loss_r_mean = loss_r_ssum / (FLAGS.logging_interval * batch_size)
                    loss_c_mean = loss_c_ssum / (FLAGS.logging_interval * batch_size)
                    loss_m_mean = loss_m_ssum / (FLAGS.logging_interval * batch_size)
                    loss_s_mean = loss_s_ssum / (FLAGS.logging_interval * batch_size)
                    loss_t_mean = loss_t_ssum / (FLAGS.logging_interval * batch_size)

                    hist_latest_str = log_sep.join(['Hist', '{:06d}',
                        '{:.9e}', '{:.9e}', '{:.9e}', '{:.9e}', '{:.9e}', '{:.9e}', '{:.9e}']).format( \
                        step_i,
                        learning_rate,
                        total_loss_mean,
                        loss_r_mean,
                        loss_c_mean,
                        loss_m_mean,
                        loss_s_mean,
                        loss_t_mean)

                    if hist_logger is None:
                        logger.info(hist_latest_str)

                    if hist_logger is not None:
                        hist_logger(step_i,
                                    learning_rate,
                                    total_loss_mean,
                                    loss_r_mean,
                                    loss_c_mean,
                                    loss_m_mean,
                                    loss_s_mean,
                                    loss_t_mean)
                        print(hist_latest_str)

                    total_loss_ssum = 0
                    loss_r_ssum = 0
                    loss_c_ssum = 0
                    loss_m_ssum = 0
                    loss_s_ssum = 0
                    loss_t_ssum = 0

                # Save checkpoint
                if step_i % checkpoint_interval == (checkpoint_interval-1) or ((step_i) == (max_steps-1)):
                    # Output Summary
                    if FLAGS.save_summary:
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, step_i)
                    #
                    checkpoint_path = os.path.join(out_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step_i)
                    trained_model_checkpoint_path = '{}-{}'.format(checkpoint_path, step_i)
                    logger.info('Model was saved at: {}'.format(trained_model_checkpoint_path))

    sess.close()
    return trained_model_checkpoint_path

def validate(dataset_frame1, dataset_frame2, dataset_frame3):
    """Performs validation on model.
    Args:
    """
    pass


def test(dataset_frame1, dataset_frame2, dataset_frame3, target_time_point=0.5, trained_model_checkpoint_path=None):
    # def rgb2gray(rgb):
    #     return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    """Perform test on a trained model."""
    with tf.Graph().as_default():
        # Create input and target placeholder.
        input_placeholder = tf.placeholder(tf.float32, shape=(None, 256, 256, 6))
        target_placeholder = tf.placeholder(tf.float32, shape=(None, 256, 256, 3))

        edge_vgg_1 = Vgg16(input_placeholder[:, :, :, :3], reuse=None)
        edge_vgg_3 = Vgg16(input_placeholder[:, :, :, 3:6], reuse=True)

        edge_1 = tf.nn.sigmoid(edge_vgg_1.fuse)
        edge_3 = tf.nn.sigmoid(edge_vgg_3.fuse)
        logger.debug('edge_1.shape: {}'.format(edge_1.shape))

        edge_1 = tf.reshape(edge_1, [-1, input_placeholder.get_shape().as_list()[1], input_placeholder.get_shape().as_list()[2], 1])
        edge_3 = tf.reshape(edge_3, [-1, input_placeholder.get_shape().as_list()[1], input_placeholder.get_shape().as_list()[2], 1])
        logger.debug('edge_1.shape: {}'.format(edge_1.shape))

        with tf.variable_scope("Cycle_DVF"):
            # Prepare model.
            model = VoxelFlowModel(batch_size=1, is_train=False)
            prediction, _ = model.inference(tf.concat([input_placeholder, edge_1, edge_3], 3),
                                            target_time_point=target_time_point)

        # Create a saver and load.
        sess = tf.Session()

        # Restore checkpoint from file.

        if trained_model_checkpoint_path is None:
            raise Exception('Trained model checkpoint is needed.')

        if trained_model_checkpoint_path:
            restorer = tf.train.Saver()
            restorer.restore(sess, trained_model_checkpoint_path)
            logger.info('Pre-trained model restored from {}'.format(trained_model_checkpoint_path))

        # Process on test dataset.
        data_list_frame1 = dataset_frame1.read_data_list_file()
        data_size = len(data_list_frame1)
        logger.info('data_size: {}'.format(data_size))

        data_list_frame2 = dataset_frame2.read_data_list_file()

        data_list_frame3 = dataset_frame3.read_data_list_file()

        i = 0
        PSNR = 0
        SSIM = 0

        use_motion_mask_png = False
        compute_metrics = False
        for id_img in range(0, data_size):
            UCF_index = data_list_frame1[id_img][:-12]
            logger.info('id: {} | UCF_index: {}'.format(id_img, UCF_index))
            # Load single data.

            batch_data_frame1 = [dataset_frame1.process_func(os.path.join('ucf101_interp_ours', ll)[:-5] + '00.png') for
                                 ll in data_list_frame1[id_img:id_img + 1]]
            if compute_metrics:
                batch_data_frame2 = [dataset_frame2.process_func(os.path.join('ucf101_interp_ours', ll)[:-5] + '01_gt.png')
                                     for ll in data_list_frame2[id_img:id_img + 1]]
            batch_data_frame3 = [dataset_frame3.process_func(os.path.join('ucf101_interp_ours', ll)[:-5] + '02.png') for
                                 ll in data_list_frame3[id_img:id_img + 1]]

            if use_motion_mask_png:
                batch_data_mask = [
                    dataset_frame3.process_func(os.path.join('motion_masks_ucf101_interp', ll)[:-11] + 'motion_mask.png')
                    for ll in data_list_frame3[id_img:id_img + 1]]

            batch_data_frame1 = np.array(batch_data_frame1)
            if compute_metrics:
                batch_data_frame2 = np.array(batch_data_frame2)
            batch_data_frame3 = np.array(batch_data_frame3)

            if use_motion_mask_png:
                batch_data_mask = (np.array(batch_data_mask) + 1.0) / 2.0

            if compute_metrics:
                feed_dict = {input_placeholder: np.concatenate((batch_data_frame1, batch_data_frame3), 3),
                             target_placeholder: batch_data_frame2}
            else:
                feed_dict = {input_placeholder: np.concatenate((batch_data_frame1, batch_data_frame3), 3)}

            # Run single step update.
            if compute_metrics:
                prediction_np, target_np, warped_img1, warped_img2 = \
                    sess.run([prediction, target_placeholder, model.warped_img1, model.warped_img2], feed_dict=feed_dict)
            else:
                prediction_np = sess.run(prediction, feed_dict=feed_dict)

            ckpt_dir, ckpt_name = os.path.split(trained_model_checkpoint_path)
            _, ckpt_dir = os.path.split(ckpt_dir)

            # out = 'ucf101_interp_ours/' + str(UCF_index) + '/frame_01_' + ckpt_dir + '_' + ckpt_name + '.png'
            ckpt_str = ckpt_dir + '_' + ckpt_name
            out = 'ucf101_interp_ours/{}/frame_01_{}.png'.format(UCF_index, ckpt_str)
            imwrite(out, prediction_np[-1, :, :, :])

            logger.info('Generated image was saved at: {}'.format(out))

            if use_motion_mask_png:
                logger.info(np.sum(batch_data_mask))

            if False: # if (not use_motion_mask_png) or np.sum(batch_data_mask) > 0:
                img_pred_mask = np.expand_dims(batch_data_mask[0], -1) * (prediction_np[-1] + 1.0) / 2.0
                img_target_mask = np.expand_dims(batch_data_mask[0], -1) * (target_np[-1] + 1.0) / 2.0
                mse = np.sum((img_pred_mask - img_target_mask) ** 2) / (3. * np.sum(batch_data_mask))
                psnr_cur = 20.0 * np.log10(1.0) - 10.0 * np.log10(mse)

                img_pred_gray = rgb2gray((prediction_np[-1] + 1.0) / 2.0)
                img_target_gray = rgb2gray((target_np[-1] + 1.0) / 2.0)
                ssim_cur = ssim(img_pred_gray, img_target_gray, data_range=1.0)

                PSNR += psnr_cur
                SSIM += ssim_cur

                i += 1
        # logger.info("Overall PSNR: %f db" % (PSNR / i))
        # logger.info("Overall SSIM: %f db" % (SSIM / i))

        sess.close()

def generate(first, second, out, target_time_point=0.5, trained_model_checkpoint_path=None):

    data_frame1 = np.expand_dims(imread(first), 0)
    data_frame3 = np.expand_dims(imread(second), 0)

    H = data_frame1.shape[1]
    W = data_frame1.shape[2]

    adatptive_H = int(np.ceil(H / 32.0) * 32.0)
    adatptive_W = int(np.ceil(W / 32.0) * 32.0)

    pad_up = int(np.ceil((adatptive_H - H) / 2.0))
    pad_bot = int(np.floor((adatptive_H - H) / 2.0))
    pad_left = int(np.ceil((adatptive_W - W) / 2.0))
    pad_right = int(np.floor((adatptive_W - W) / 2.0))

    logger.info('input image shape: ({}, {}) -> ({}, {})'.format(H, W, adatptive_H, adatptive_W))

    """Perform test on a trained model."""
    with tf.Graph().as_default():
        # Create input and target placeholder.
        input_placeholder = tf.placeholder(tf.float32, shape=(None, H, W, 6))

        input_pad = tf.pad(input_placeholder, [[0, 0], [pad_up, pad_bot], [pad_left, pad_right], [0, 0]], 'SYMMETRIC')

        edge_vgg_1 = Vgg16(input_pad[:, :, :, :3], reuse=None)
        edge_vgg_3 = Vgg16(input_pad[:, :, :, 3:6], reuse=True)

        edge_1 = tf.nn.sigmoid(edge_vgg_1.fuse)
        edge_3 = tf.nn.sigmoid(edge_vgg_3.fuse)

        edge_1 = tf.reshape(edge_1, [-1, input_pad.get_shape().as_list()[1], input_pad.get_shape().as_list()[2], 1])
        edge_3 = tf.reshape(edge_3, [-1, input_pad.get_shape().as_list()[1], input_pad.get_shape().as_list()[2], 1])

        with tf.variable_scope("Cycle_DVF"):
            # Prepare model.
            model = VoxelFlowModel(batch_size=1, is_train=False)
            prediction, _ = model.inference(tf.concat([input_pad, edge_1, edge_3], 3),
                                            target_time_point=target_time_point)

        # Create a saver and load.
        with tf.Session() as sess:

            # Restore checkpoint from file.
            if trained_model_checkpoint_path is None:
                raise Exception('Trained model checkpoint is not available.')

            if trained_model_checkpoint_path:
                restorer = tf.train.Saver()
                restorer.restore(sess, trained_model_checkpoint_path)
                logger.info('Pre-trained model restored from: {}'.format(trained_model_checkpoint_path))

            feed_dict = {input_placeholder: np.concatenate((data_frame1, data_frame3), 3)}
            # Run single step update.
            prediction_np = sess.run(prediction, feed_dict=feed_dict)

            img_output = prediction_np[-1, pad_up:adatptive_H - pad_bot, pad_left:adatptive_W - pad_right, :]

            imwrite(out, img_output)
            logger.info('Generated image was saved at: {}'.format(out))



hist_logging = False
try:
    from table_logger import TableLogger
    hist_logging = True
except:
    print('Continue running without logging to a CSV file as table-logger is not installed.'
          ' To enable logging, "pip install table-logger" and rerun this code.')


def timestamp():
    dt = datetime.now()
    dt += timedelta(hours=8) # timezone('Asia/Singapore')
    return dt.strftime('%Y-%m-%dT%H%M%S')

def insert_str(string, index, str_to_insert):
    return string[:index] + str_to_insert + string[index:]

if __name__ == '__main__':

    start_time = timestamp()

    config_str = '_'.join([start_time])
    out_dir = FLAGS.train_dir + '/' + config_str
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    log_sep = ' ,'

    hist_logger = None
    if hist_logging:
        history_cols = ['Step', 'Learning_Rate', 'Total_Loss', 'Reconstruction_Loss', 'Cycle_Consistency_Loss',
                        'Motion_Linearity_Loss', 'Stable_Motion_Loss', 'Temporal_Regularization_Loss']
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

    if FLAGS.logging_level.lower() == 'debug':
        logging_level = logging.DEBUG
    if FLAGS.logging_level.lower() == 'info':
        logging_level = logging.INFO
    if FLAGS.logging_level.lower() == 'warn':
        logging_level = logging.WARN

    logging.basicConfig(level=logging_level,
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
        logger.info('task: {}'.format(FLAGS.task))
        logger.info('pretrained_model_checkpoint_path: {}'.format(FLAGS.pretrained_model_checkpoint_path))
        logger.info('batch_size: {}'.format(FLAGS.batch_size))
        logger.info('training_data_step: {}'.format(FLAGS.training_data_step))
        logger.info('model_size: {}'.format(FLAGS.model_size))
        logger.info('dataset_train: {}'.format(FLAGS.dataset_train))
        logger.info('dataset_test: {}'.format(FLAGS.dataset_test))
        logger.info('stage: {}'.format(FLAGS.stage))
        logger.info('max_steps: {}'.format(FLAGS.max_steps))
        logger.info('max_epochs: {}'.format(FLAGS.max_epochs))
        logger.info('s1_steps: {}'.format(FLAGS.s1_steps))
        logger.info('s1_epochs: {}'.format(FLAGS.s1_epochs))
        logger.info('crop_size: {}'.format(FLAGS.crop_size))
        logger.info('coef_loss_c: {}'.format(FLAGS.coef_loss_c))
        logger.info('coef_loss_m: {}'.format(FLAGS.coef_loss_m))
        logger.info('coef_loss_s: {}'.format(FLAGS.coef_loss_s))
        logger.info('coef_loss_t: {}'.format(FLAGS.coef_loss_t))
        logger.info('strategy: {}'.format(FLAGS.strategy))
        logger.info('logging_interval: {}'.format(FLAGS.logging_interval))
        logger.info('checkpoint_interval: {}'.format(FLAGS.checkpoint_interval))
        logger.info('save_summary: {}'.format(FLAGS.save_summary))
        logger.info('graph_level_seed: {}'.format(FLAGS.graph_level_seed))
        logger.info('logging_level: {}'.format(FLAGS.logging_level))

        logger.info('Output_directory: {}'.format(out_dir))

        if FLAGS.model_size != 'large':
            # from CyclicGen_model import VoxelFlowModel
            logger.error('Only large model is supported.')

        trained_model_checkpoint_path = FLAGS.pretrained_model_checkpoint_path

        if 'train' in FLAGS.task.lower():
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

            if FLAGS.dataset_train == 'ucf101':
                data_list_path_frame1 = "data_list/ucf101_train_files_frame1.txt"
                data_list_path_frame2 = "data_list/ucf101_train_files_frame2.txt"
                data_list_path_frame3 = "data_list/ucf101_train_files_frame3.txt"
            if FLAGS.dataset_train == 'ucf101_256':
                data_list_path_frame1 = "data_list/ucf101_256_train_files_frame1.txt"
                data_list_path_frame2 = "data_list/ucf101_256_train_files_frame2.txt"
                data_list_path_frame3 = "data_list/ucf101_256_train_files_frame3.txt"
            if FLAGS.dataset_train == 'middlebury':
                data_list_path_frame1 = "data_list/middlebury_train_files_frame1.txt"
                data_list_path_frame2 = "data_list/middlebury_train_files_frame2.txt"
                data_list_path_frame3 = "data_list/middlebury_train_files_frame3.txt"

            ucf101_dataset_frame1 = Dataset(data_list_path_frame1)
            ucf101_dataset_frame2 = Dataset(data_list_path_frame2)
            ucf101_dataset_frame3 = Dataset(data_list_path_frame3)

            trained_model_checkpoint_path = \
                train(ucf101_dataset_frame1, ucf101_dataset_frame2, ucf101_dataset_frame3, out_dir,
                      log_sep=' ,', hist_logger=hist_logger,
                      trained_model_checkpoint_path=trained_model_checkpoint_path)

        if 'test' in FLAGS.task.lower():
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

            if FLAGS.dataset_test == 'ucf101':
                data_list_path_frame1 = "data_list/ucf101_test_files_frame1.txt"
                data_list_path_frame2 = "data_list/ucf101_test_files_frame2.txt"
                data_list_path_frame3 = "data_list/ucf101_test_files_frame3.txt"
            if FLAGS.dataset_test == 'ucf101_256':
                data_list_path_frame1 = "data_list/ucf101_256_train_files_frame1.txt"
                data_list_path_frame2 = "data_list/ucf101_256_train_files_frame2.txt"
                data_list_path_frame3 = "data_list/ucf101_256_train_files_frame3.txt"
            if FLAGS.dataset_test == 'middlebury':
                data_list_path_frame1 = "data_list/middlebury_test_files_frame1.txt"
                data_list_path_frame2 = "data_list/middlebury_test_files_frame2.txt"
                data_list_path_frame3 = "data_list/middlebury_test_files_frame3.txt"

            ucf101_dataset_frame1 = Dataset(data_list_path_frame1)
            ucf101_dataset_frame2 = Dataset(data_list_path_frame2)
            ucf101_dataset_frame3 = Dataset(data_list_path_frame3)

            test(ucf101_dataset_frame1, ucf101_dataset_frame2, ucf101_dataset_frame3,
                 trained_model_checkpoint_path=trained_model_checkpoint_path)

        if 'generate' in FLAGS.task.lower():
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

            if True: #if FLAGS.dataset_test == 'middlebury':
                data_list_path_frame1 = "data_list/middlebury_test_files_frame1.txt"
                data_list_path_frame2 = "data_list/middlebury_test_files_frame2.txt"
                data_list_path_frame3 = "data_list/middlebury_test_files_frame3.txt"

            ucf101_dataset_frame1 = Dataset(data_list_path_frame1)
            ucf101_dataset_frame2 = Dataset(data_list_path_frame2)
            ucf101_dataset_frame3 = Dataset(data_list_path_frame3)

            for tp in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.5]:
                gen_i0_path = FLAGS.gen_i0_path # './Middlebury/eval-color-allframes/eval-data/Backyard/frame07.png'
                gen_i1_path = FLAGS.gen_i1_path # './Middlebury/eval-color-allframes/eval-data/Backyard/frame08.png'

                i0_num = gen_i0_path[-len('.png') - 2:-len('.png')]
                i1_num = gen_i1_path[-len('.png') - 2:-len('.png')]

                gen_out_path = FLAGS.gen_out_path
                if gen_out_path is None:
                    ckpt_dir, ckpt_name = os.path.split(trained_model_checkpoint_path)
                    _, ckpt_dir = os.path.split(ckpt_dir)

                    gen_out_path = gen_i0_path[:-len('07.png')] + \
                                   '{}_from{}_{}_{}.png'.format(insert_str('{:05.2f}'.format(int(i0_num) + tp), 2, '_'),
                                                            i0_num,
                                                            i1_num,
                                                            (ckpt_dir + '_' + ckpt_name))

                generate(gen_i0_path, gen_i1_path, gen_out_path, target_time_point=tp,
                         trained_model_checkpoint_path=trained_model_checkpoint_path)

    except:
        logger.exception('### An Exception occurred.')