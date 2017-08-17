import tensorflow as tf
import Tools
from DataSet import OneFlodDataSet
import numpy as np
from trained_vgg16 import vgg16
from tools import calculate_features
from SVR import MySVR
import copy


def detection_image(image_path, type='cross'):
    flod_dir = '/home/give/Documents/dataset/FaceDetection/Bounding_Box/train'
    test_image_path = image_path
    # test_image_path = '/home/give/PycharmProjects/FaceDetection/fine_tuning_vgg16/IMGP0857.jpg'
    test_images, regions = Tools.create_image_selected_rectangle(test_image_path)
    regions_copy = copy.copy(regions)
    print 'test image shape is ', np.shape(test_images)
    threshold = 0.3
    dataset = OneFlodDataSet(flod_dir, threshold)
    train_images, train_labels, train_scores = dataset.next_batch(2048, [1024, 1024])
    validation_images, validation_labels, validation_scores = dataset.next_batch(128, [96, 32])
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
    vgg = vgg16(imgs, '/home/give/PycharmProjects/FaceDetection/fine_tuning_vgg16/vgg16_trained.npy', sess, True)
    train_features = calculate_features(train_images, sess, vgg.feature, vgg.imgs)
    svr = MySVR(train_features, train_labels, train_scores, None, validation_labels, validation_scores)

    # ---start test-------
    test_features = calculate_features(test_images, sess, vgg.feature, vgg.imgs)
    test_labels = calculate_features(test_images, sess, tf.argmax(vgg.y, 1), vgg.imgs)
    test_scores = svr.do_predict(test_features)
    test_scores[test_labels == 0] = 0.0
    test_scoress = np.array(test_scores)
    regions_copy = np.array(regions_copy)
    EQUAL_THRESHOLD = 0.5
    index = [test_scoress >= EQUAL_THRESHOLD]
    test_scoress = test_scoress[index]
    regions_copy = regions_copy[index]
    Tools.non_maximum_suppression(test_image_path, test_scoress, regions_copy, type)

if __name__ == '__main__':
    # image_path = '/home/give/PycharmProjects/FaceDetection/fine_tuning_vgg16/timg.jpg'
    image_path = '/home/give/Documents/dataset/FaceDetection/TenFolds/02/img_4.jpg'
    detection_image(image_path, 'cross')