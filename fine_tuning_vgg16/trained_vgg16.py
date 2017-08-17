########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import tensorflow as tf
import numpy as np
from tools import do_conv, pool, FC_layer, batch_norm, load_with_skip


class vgg16:
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.trainable = {
            'conv1_1': False,
            'conv1_2': False,
            'conv2_1': False,
            'conv2_2': False,
            'conv3_1': False,
            'conv3_2': False,
            'conv3_3': False,
            'conv4_1': False,
            'conv4_2': False,
            'conv4_3': False,
            'conv5_1': False,
            'conv5_2': False,
            'conv5_3': False,
            'fc6': True,
            'fc7': True,
            'fc8': True
        }
        self.layers_name = [
            'conv1_1',
            'conv1_2',
            'conv2_1',
            'conv2_2',
            'conv3_1',
            'conv3_2',
            'conv3_3',
            'conv4_1',
            'conv4_2',
            'conv4_3',
            'conv5_1',
            'conv5_2',
            'conv5_3',
            'fc6',
            'fc7',
            'fc8'
        ]
        self.classesnumber = 2
        self.regularizer = None
        self.convlayers()
        self.fc_layers()
        self.y = self.fcs_output
        # self.feature = self.fc2
        if weights is not None and sess is not None:
            load_with_skip(weights, sess, ['fc8'])


    def convlayers(self):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean
        # images = self.imgs
        # conv1_1
        # do_conv(name, input_tensor, out_channel, ksize, stride=[1, 1, 1, 1], is_pretrain=True):
        images = do_conv('conv1_1', images, 64, [3, 3], is_pretrain=self.trainable['conv1_1'], regularizer=self.regularizer)
        images = do_conv('conv1_2', images, 64, [3, 3], is_pretrain=self.trainable['conv1_2'], regularizer=self.regularizer)

        images = pool('pooling1', images, is_max_pool=True)

        images = do_conv('conv2_1', images, 128, [3, 3], is_pretrain=self.trainable['conv2_1'], regularizer=self.regularizer)
        images = do_conv('conv2_2', images, 128, [3, 3], is_pretrain=self.trainable['conv2_2'], regularizer=self.regularizer)

        images = pool('pooling2', images, is_max_pool=True)

        images = do_conv('conv3_1', images, 256, [3, 3], is_pretrain=self.trainable['conv3_1'], regularizer=self.regularizer)
        images = do_conv('conv3_2', images, 256, [3, 3], is_pretrain=self.trainable['conv3_2'], regularizer=self.regularizer)
        images = do_conv('conv3_3', images, 256, [3, 3], is_pretrain=self.trainable['conv3_3'], regularizer=self.regularizer)

        images = pool('pooing3', images, is_max_pool=True)

        images = do_conv('conv4_1', images, 512, [3, 3], is_pretrain=self.trainable['conv4_1'], regularizer=self.regularizer)
        images = do_conv('conv4_2', images, 512, [3, 3], is_pretrain=self.trainable['conv4_2'], regularizer=self.regularizer)
        images = do_conv('conv4_3', images, 512, [3, 3], is_pretrain=self.trainable['conv4_3'], regularizer=self.regularizer)

        images = pool('pooling4', images, is_max_pool=True)

        images = do_conv('conv5_1', images, 512, [3, 3], is_pretrain=self.trainable['conv5_1'], regularizer=self.regularizer)
        images = do_conv('conv5_2', images, 512, [3, 3], is_pretrain=self.trainable['conv5_2'], regularizer=self.regularizer)
        images = do_conv('conv5_3', images, 512, [3, 3], is_pretrain=self.trainable['conv5_3'], regularizer=self.regularizer)

        images = pool('pooling5', images, is_max_pool=True)

        self.convs_output = images

    def fc_layers(self):
        # def FC_layer(layer_name, x, out_nodes):
        images = FC_layer('fc6', self.convs_output, 4096, regularizer=self.regularizer)
        images = batch_norm(images)
        images = FC_layer('fc7', images, 4096, regularizer=self.regularizer)
        images = batch_norm(images)
        self.feature = images
        images = FC_layer('fc8', images, self.classesnumber, regularizer=self.regularizer)

        self.fcs_output = images

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print i, k, np.shape(weights[k])
            sess.run(self.parameters[i].assign(weights[k]))