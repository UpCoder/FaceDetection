# -*- coding: utf-8 -*-
import tensorflow as tf
import VGG16
import LeNet
import numpy as np
import gc
import Tools
import CreateNPY
from DataSet import OneFlodDataSet
EQUAL_THRESHOLD = 0.3
IMAGE_W = 224
IMAGE_H = 224
IMAGE_CHANNAL = 3
OUTPUT_NODE = 2
LEARNING_RATE = 1e-3
MAX_ITERATION = int(1e+5)
BATCH_SZIE = 256
BATCH_DISTRIBUTION = [128, 128]


def train(dataset):
    x = tf.placeholder(
        tf.float32,
        shape=[
            BATCH_SZIE,
            IMAGE_W,
            IMAGE_H,
            IMAGE_CHANNAL
        ],
        name='input_x'
    )
    y_ = tf.placeholder(
        tf.float32,
        shape=[
            BATCH_SZIE,
            OUTPUT_NODE
        ],
        name='input_y'
    )
    feature, y = VGG16.inference(x, None)
    # y = LeNet.interfernece(x, True, None)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=y,
            labels=y_
        )
    )
    tf.summary.scalar(
        'loss',
        loss
    )
    train_op = tf.train.GradientDescentOptimizer(
        learning_rate=LEARNING_RATE
    ).minimize(loss)
    # 计算准确率
    with tf.name_scope('accuracy'):
        correct_predict = tf.equal(
            tf.argmax(y, 1),
            tf.argmax(y_, 1)
        )
        accuracy_tensor = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
        tf.summary.scalar(
            'accuracy',
            accuracy_tensor
        )
    tf.summary.histogram(
        'labels',
        tf.argmax(y_, 1)
    )
    tf.summary.histogram(
        'logits',
        tf.argmax(y, 1)
    )
    merged = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        log_path = './log'
        writer = tf.summary.FileWriter(log_path, tf.get_default_graph())
        for i in range(MAX_ITERATION):
            train_images, labels, scores = dataset.next_batch(BATCH_SZIE, BATCH_DISTRIBUTION)
            feed_dict = {
                x: train_images,
                y_: labels
            }
            _, loss_value, accuracy_value, summary, y_value = sess.run(
                [train_op, loss, accuracy_tensor, merged, y],
                feed_dict=feed_dict
            )
            writer.add_summary(summary, i)
            if (i % 20) == 0:
                print 'predict y value is ', np.argmax(y_value)
                print 'loss value is %g accuracy is %g' \
                      % (loss_value, accuracy_value)
            del train_images, labels, scores
            gc.collect()


if __name__ == '__main__':
    # ANNOTATION_FILES_DIR = '/home/give/Documents/dataset/FaceDetection/FDDB-folds'
    # annotation_dict = CreateNPY.create_annotation_dict(ANNOTATION_FILES_DIR)
    flod_dir = '/home/give/Documents/dataset/FaceDetection/Bounding_Box/01'
    threshold = 0.3
    dataset = OneFlodDataSet(flod_dir, threshold)
    train(dataset)
