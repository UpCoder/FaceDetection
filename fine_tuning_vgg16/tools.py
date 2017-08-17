import tensorflow as tf
import numpy as np


def do_conv(name, input_tensor, out_channel, ksize, stride=[1, 1, 1, 1], is_pretrain=True, regularizer=None):
    input_channel = input_tensor.get_shape()[-1]
    with tf.variable_scope(name):
        weights = tf.get_variable(
            'weights',
            shape=[
                ksize[0],
                ksize[1],
                input_channel,
                out_channel,
            ],
            trainable=is_pretrain,
            initializer=tf.contrib.layers.xavier_initializer()
        )
        bias = tf.get_variable(
            'biases',
            shape=[
                out_channel
            ],
            trainable=is_pretrain,
            initializer=tf.constant_initializer(0.0)
        )
        x = tf.nn.conv2d(
            input_tensor,
            weights,
            strides=stride,
            padding='SAME',
            name='conv'
        )
        x = tf.nn.bias_add(
            x, bias,
            name='bias-add'
        )
        x = tf.nn.relu(x, name='relu')
        return x


def pool(layer_name, x, kernel=[1,2,2,1], stride=[1, 2, 2, 1], is_max_pool=True):
    '''Pooling op
    Args:
        x: input tensor
        kernel: pooling kernel, VGG paper used [1,2,2,1], the size of kernel is 2X2
        stride: stride size, VGG paper used [1,2,2,1]
        padding:
        is_max_pool: boolen
                    if True: use max pooling
                    else: use avg pooling
    '''
    if is_max_pool:
        x = tf.nn.max_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    else:
        x = tf.nn.avg_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    return x


def batch_norm(x):
    '''Batch normlization(I didn't include the offset and scale)
    '''
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x, [0])
    x = tf.nn.batch_normalization(x,
                                  mean=batch_mean,
                                  variance=batch_var,
                                  offset=None,
                                  scale=None,
                                  variance_epsilon=epsilon)
    return x


def FC_layer(layer_name, x, out_nodes, regularizer=None):
    '''Wrapper for fully connected layers with RELU activation as default
    Args:
        layer_name: e.g. 'FC1', 'FC2'
        x: input feature map
        out_nodes: number of neurons for current FC layer
    '''
    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights',
                            shape=[size, out_nodes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('biases',
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))
        if regularizer is not None:
            tf.contrib.layers.apply_regularization(regularizer, [w])
        flat_x = tf.reshape(x, [-1, size])  # flatten into 1D
        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        x = tf.nn.relu(x)
        return x


def calculate_loss(logits, labels):
    '''Compute loss
    Args:
        logits: logits tensor, [batch_size, n_classes]
        labels: one-hot labels
    '''
    with tf.name_scope('loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross-entropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        # loss_train = tf.add_n(tf.get_collection('losses'))
        # print loss_train
        tf.summary.scalar(scope+'/loss', loss)
        return loss


def calculate_accuracy(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor,
    """
    with tf.name_scope('accuracy') as scope:
        correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope+'/accuracy', accuracy)
    return accuracy


def optimize(loss, learning_rate, global_step):
    '''optimization, use Gradient Descent as default
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op


def load(data_path, session):
    data_dict = np.load(data_path, encoding='latin1').item()

    keys = sorted(data_dict.keys())
    for key in keys:
        with tf.variable_scope(key, reuse=True):
            for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                session.run(tf.get_variable(subkey).assign(data))


def load_with_skip(data_path, session, skip_layer):
    print data_path
    data_dict = np.load(data_path, encoding='latin1').item()
    for key in data_dict:
        print key
        if key not in skip_layer:
            with tf.variable_scope(key, reuse=True):
                for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                    session.run(tf.get_variable(subkey).assign(data))


def save_weights(save_path, names):
    res_dict = {}
    for name in names:
        # get the variable
        with tf.variable_scope(name, reuse=True):
            res_dict[name] = []
            res_dict[name].append(tf.get_variable('weights').eval())
            res_dict[name].append(tf.get_variable('biases').eval())
    np.save(
        save_path,
        res_dict
    )
    print 'save successful', save_path


def calculate_features(images, sess, cal_tensor, input_tensor):
    length = len(images)
    batch_size = 100
    features = []
    index = 0
    while index < length:
        end_index = index + batch_size
        if end_index >= length:
            end_index = length
        batch_images = images[index:end_index]
        train_features = sess.run(
            cal_tensor,
            feed_dict={
                input_tensor: batch_images
            }
        )
        features.extend(train_features)
        index += batch_size
    print 'features shape is ', np.shape(features)
    return np.array(features)
