# -*- coding: utf-8 -*-
import tensorflow as tf
from trained_vgg16 import vgg16
from DataSet import OneFlodDataSet
import numpy as np
import gc
import Tools


class train:
    def __init__(self):
        flod_dir = '/home/give/Documents/dataset/FaceDetection/Bounding_Box/01'
        threshold = 0.3
        self.dataset = OneFlodDataSet(flod_dir, threshold)
        sess = tf.Session()
        imgs = tf.placeholder(
            tf.float32,
            shape=[
                None,
                224,
                224,
                3
            ]
        )
        self.vgg = vgg16(imgs, 'vgg16_weights.npz', sess)
        self.pooling5 = self.vgg.pool5
        self.iterator_number = int(1e+5)
        self.learning_rate = 0.1
        self.BATCH_SZIE = 100
        self.BATCH_DISTRIBUTION = [68, 32]

    def inference(self):
        # fc1
        with tf.name_scope('train_fc1') as scope:
            shape = int(np.prod(self.pooling5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                               trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pooling5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)

        # fc2
        with tf.name_scope('train_fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                               trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)

        # fc3
        with tf.name_scope('train_fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([4096, 2],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[2], dtype=tf.float32),
                               trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
        return self.fc3l

    def start_train(self):
        y_ = tf.placeholder(
            tf.float32,
            [
                None,
                2
            ]
        )
        y = self.inference()
        y = tf.nn.softmax(y)
        loss = -tf.reduce_mean(
            y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
        )
        # loss = tf.reduce_mean(
        #     tf.nn.sparse_softmax_cross_entropy_with_logits(
        #         logits=y,
        #         labels=tf.argmax(y_, 1)
        #     )
        # )
        tf.summary.scalar(
            'loss',
            loss
        )
        train_op = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate
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
        merged = tf.summary.merge_all()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            log_path = './log'
            writer = tf.summary.FileWriter(log_path, tf.get_default_graph())
            for i in range(self.iterator_number):
                train_images, labels, scores = self.dataset.next_batch(self.BATCH_SZIE, self.BATCH_DISTRIBUTION)
                feed_dict = {
                    self.vgg.imgs: train_images,
                    y_: labels
                }
                _, loss_value, accuracy_value, summary, y_value = sess.run(
                    [train_op, loss, accuracy_tensor, merged, y],
                    feed_dict=feed_dict
                )
                writer.add_summary(summary, i)
                if (i % 20) == 0:
                    print 'predict y value is ', np.argmax(y_value, 1)
                    print 'loss value is %g accuracy is %g' \
                          % (loss_value, accuracy_value)
                del train_images, labels, scores
                gc.collect()


if __name__ == '__main__':
    my_train = train()
    my_train.start_train()