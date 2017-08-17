# -*- coding: utf-8 -*-
# 上采样之后得到的得分，然后通过argmax来得到最后的分类结果
import tensorflow as tf
import numpy as np
import gc

IMAGE_CHANNAL = 3

CONV1_1_SIZE = 3
CONV1_1_DEEP = 64
CONV1_2_SIZE = 3
CONV1_2_DEEP = 64

CONV2_1_SIZE = 3
CONV2_1_DEEP = 128
CONV2_2_SIZE = 3
CONV2_2_DEEP = 128

CONV3_1_SIZE = 3
CONV3_1_DEEP = 256
CONV3_2_SIZE = 3
CONV3_2_DEEP = 256
CONV3_3_SIZE = 1
CONV3_3_DEEP = 256

CONV4_1_SIZE = 3
CONV4_1_DEEP = 512
CONV4_2_SIZE = 3
CONV4_2_DEEP = 512
CONV4_3_SIZE = 1
CONV4_3_DEEP = 512

CONV5_1_SIZE = 3
CONV5_1_DEEP = 512
CONV5_2_SIZE = 3
CONV5_2_DEEP = 512
CONV5_3_SIZE = 1
CONV5_3_DEEP = 512

FC1_NODE = 4096
FC2_NODE = 4096
FC3_NODE = 2

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "50", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "'/home/give/Documents/dataset/ADEChallengeData2016'", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-1", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")
MAX_ITERATION = int(1e5 + 1)
DECAY_LEARNING_RATE = 0.1


def do_conv(name, weight_shape, bias_shape, input_tensor):
    with tf.variable_scope(name):
        weight = tf.get_variable(
            'weight',
            shape=weight_shape,
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        bias = tf.get_variable(
            'bias',
            shape=bias_shape,
            initializer=tf.constant_initializer(0.0)
        )
        conv = tf.nn.conv2d(
            input_tensor,
            weight,
            strides=[1, 1, 1, 1],
            padding='SAME',
        )
        layer = tf.nn.bias_add(conv, bias)
        return tf.nn.relu(layer)


def do_full_connection(name, weight_shape, bias_shape, input_tensor):
    with tf.variable_scope(name):
        weight = tf.get_variable(
            'weight',
            shape=weight_shape,
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        bias = tf.get_variable(
            'bias',
            shape=bias_shape,
            initializer=tf.constant_initializer(0.0)
        )
        matmul = tf.matmul(input_tensor, weight) + bias
        return tf.nn.relu(matmul)


def inference(image, keep_prob):
    with tf.variable_scope('inference'):

        layer11 = do_conv(
            'conv1_1',
            weight_shape=[
                CONV1_1_SIZE,
                CONV1_1_SIZE,
                IMAGE_CHANNAL,
                CONV1_1_DEEP
            ],
            bias_shape=[
                CONV1_1_DEEP
            ],
            input_tensor=image
        )
        print layer11.shape
        layer12 = do_conv(
            'conv1_2',
            weight_shape=[
                CONV1_2_SIZE,
                CONV1_2_SIZE,
                CONV1_1_DEEP,
                CONV1_2_DEEP
            ],
            bias_shape=[
                CONV1_2_DEEP
            ],
            input_tensor=layer11
        )
        print layer12.shape

        with tf.variable_scope('pooling1'):
            pooling1 = tf.nn.max_pool(
                layer12,
                strides=[1, 2, 2, 1],
                padding='SAME',
                ksize=[1, 2, 2, 1]
            )
            print pooling1.shape

        layer21 = do_conv(
            'conv2_1',
            weight_shape=[
                CONV2_1_SIZE,
                CONV2_1_SIZE,
                CONV1_2_DEEP,
                CONV2_1_DEEP
            ],
            bias_shape=[
                CONV2_1_DEEP
            ],
            input_tensor=pooling1
        )
        layer22 = do_conv(
            'conv2_2',
            weight_shape=[
                CONV2_2_SIZE,
                CONV2_2_SIZE,
                CONV2_1_DEEP,
                CONV2_2_DEEP
            ],
            bias_shape=[
                CONV2_2_DEEP
            ],
            input_tensor=layer21
        )

        with tf.variable_scope('pooling2'):
            pooling2 = tf.nn.max_pool(
                layer22,
                strides=[1, 2, 2, 1],
                padding='SAME',
                ksize=[1, 2, 2, 1]
            )
            print pooling2.shape

        layer31 = do_conv(
            'conv3_1',
            weight_shape=[
                CONV3_1_SIZE,
                CONV3_1_SIZE,
                CONV2_2_DEEP,
                CONV3_1_DEEP
            ],
            bias_shape=[
                CONV3_1_DEEP
            ],
            input_tensor=pooling2
        )
        layer32 = do_conv(
            'conv3_2',
            weight_shape=[
                CONV3_2_SIZE,
                CONV3_2_SIZE,
                CONV3_1_DEEP,
                CONV3_2_DEEP
            ],
            bias_shape=[
                CONV3_2_DEEP
            ],
            input_tensor=layer31
        )
        layer33 = do_conv(
            'conv3_3',
            weight_shape=[
                CONV3_3_SIZE,
                CONV3_3_SIZE,
                CONV3_2_DEEP,
                CONV3_3_DEEP
            ],
            bias_shape=[
                CONV3_3_DEEP
            ],
            input_tensor=layer32
        )
        with tf.variable_scope('pooling3'):
            pooling3 = tf.nn.max_pool(
                layer33,
                strides=[1, 2, 2, 1],
                padding='SAME',
                ksize=[1, 2, 2, 1]
            )
            print pooling3.shape
        layer41 = do_conv(
            'conv4_1',
            weight_shape=[
                CONV4_1_SIZE,
                CONV4_1_SIZE,
                CONV3_3_DEEP,
                CONV4_1_DEEP
            ],
            bias_shape=[
                CONV4_1_DEEP
            ],
            input_tensor=pooling3
        )
        layer42 = do_conv(
            'conv4_2',
            weight_shape=[
                CONV4_2_SIZE,
                CONV4_2_SIZE,
                CONV4_1_DEEP,
                CONV4_2_DEEP
            ],
            bias_shape=[
                CONV4_2_DEEP
            ],
            input_tensor=layer41
        )
        layer43 = do_conv(
            'conv4_3',
            weight_shape=[
                CONV4_3_SIZE,
                CONV4_3_SIZE,
                CONV4_2_DEEP,
                CONV4_3_DEEP
            ],
            bias_shape=[
                CONV4_3_DEEP
            ],
            input_tensor=layer42
        )
        with tf.variable_scope('pooling4'):
            pooling4 = tf.nn.max_pool(
                layer43,
                strides=[1, 2, 2, 1],
                padding='SAME',
                ksize=[1, 2, 2, 1]
            )
            print pooling4.shape
        layer51 = do_conv(
            'conv5_1',
            weight_shape=[
                CONV5_1_SIZE,
                CONV5_1_SIZE,
                CONV4_3_DEEP,
                CONV5_1_DEEP
            ],
            bias_shape=[
                CONV5_1_DEEP
            ],
            input_tensor=pooling4
        )
        layer52 = do_conv(
            'conv5_2',
            weight_shape=[
                CONV5_2_SIZE,
                CONV5_2_SIZE,
                CONV5_1_DEEP,
                CONV5_2_DEEP
            ],
            bias_shape=[
                CONV5_2_DEEP
            ],
            input_tensor=layer51
        )
        layer53 = do_conv(
            'conv5_3',
            weight_shape=[
                CONV5_3_SIZE,
                CONV5_3_SIZE,
                CONV5_2_DEEP,
                CONV5_3_DEEP
            ],
            bias_shape=[
                CONV5_3_DEEP
            ],
            input_tensor=layer52
        )
        with tf.variable_scope('pooling5'):
            pooling5 = tf.nn.max_pool(
                layer53,
                strides=[1, 2, 2, 1],
                padding='SAME',
                ksize=[1, 2, 2, 1]
            )
            print pooling5.shape

        shape = pooling5.get_shape().as_list()
        nodes = shape[1] * shape[2] * shape[3]
        reshaped = tf.reshape(
            pooling5,
            [
                -1,
                nodes
            ]
        )
        fc1_result = do_full_connection('full_connection1', [nodes, FC1_NODE], [FC1_NODE], reshaped)
        fc2_result = do_full_connection('full_connection2', [FC1_NODE, FC2_NODE], [FC2_NODE], fc1_result)
        fc3_result = do_full_connection('full_connection3', [FC2_NODE, FC3_NODE], [FC3_NODE], fc2_result)
        return fc2_result, fc3_result


if __name__ == '__main__':
    test_tensor = tf.placeholder(
        tf.float32,
        [
            50,
            224,
            224,
            3
        ],
        name='input'
    )
    fc2_result, result = inference(test_tensor, None)
    print result