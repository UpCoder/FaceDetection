# -*- coding: utf-8 -*-
import tensorflow as tf

# 输入输出节点的个数
OUTPUT_NODE = 2

IMAGE_SIZE = 128
NUM_CHANNELS = 3
NUM_LABELS = 2

# 第一层卷积的大小以及深度
CONV1_DEEP = 32
CONV1_SIZE = 5


# 第二层卷积的大小以及深度
CONV2_DEEP = 64
CONV2_SIZE = 5

# 全连接层的节点个数
FC_SIZE = 512

def interfernece(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weight = tf.get_variable(
            'weight',
            [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv1_baises = tf.get_variable(
            'bias',
            [CONV1_DEEP],
            initializer=tf.constant_initializer(0.0)
        )
        conv1 = tf.nn.conv2d(
            input_tensor,
            conv1_weight,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )
        relu1 = tf.nn.relu(
            tf.nn.bias_add(
                conv1, conv1_baises)
        )

    with tf.variable_scope('layer2-maxpooling'):
        pool1 = tf.nn.max_pool(
            relu1,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME'
        )

    with tf.variable_scope('layer3-conv2'):
        conv2_weight = tf.get_variable(
            'weight',
            [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )

        conv2_baises = tf.get_variable(
            'bais',
            [CONV2_DEEP],
            initializer=tf.constant_initializer(0.0)
        )

        conv2 = tf.nn.conv2d(
            pool1,
            conv2_weight,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )

        relu2 = tf.nn.relu(
            tf.nn.bias_add(conv2, conv2_baises)
        )

    with tf.variable_scope('layer4-maxpooling2'):
        pool2 = tf.nn.max_pool(
            relu2,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME'
        )

    # 这个数组一共有四个维度
    # 第一个维度是batch的个数
    # 第二个和第三个维度是大小
    # 第四个维度是深度
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

    # 将一个四维的数组变成二维的
    reshaped = tf.reshape(pool2, [-1, nodes])

    with tf.variable_scope('layer5-fc1'):
        fc1_weight = tf.get_variable(
            'weight',
            [nodes, FC_SIZE],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        if regularizer is not None:
            tf.add_to_collection(
                'losses', regularizer(fc1_weight)
            )

        fc1_baises = tf.get_variable(
            'bias',
            [FC_SIZE],
            initializer=tf.constant_initializer(0.1)
        )
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weight) + fc1_baises)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc2'):
        fc2_weight = tf.get_variable(
            'weight',
            [FC_SIZE, OUTPUT_NODE],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        if regularizer is not None:
            tf.add_to_collection(
                'losses', regularizer(fc2_weight)
            )
        fc2_baises = tf.get_variable(
            'bais',
            [OUTPUT_NODE],
            initializer=tf.constant_initializer(0.1)
        )
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weight) + fc2_baises)
    return fc2