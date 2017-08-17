import tensorflow as tf
import Tools
from DataSet import OneFlodDataSet
import numpy as np
from trained_vgg16 import vgg16
from SVR import MySVR
import copy


def detection_image(image_path, type='cross'):
    flod_dir = '/home/give/Documents/dataset/FaceDetection/Bounding_Box/01'
    test_image_path = image_path
    # test_image_path = '/home/give/PycharmProjects/FaceDetection/fine_tuning_vgg16/IMGP0857.jpg'
    test_images, regions = Tools.create_image_selected_rectangle(test_image_path)
    regions_copy = copy.copy(regions)
    print 'test image shape is ', np.shape(test_images)
    threshold = 0.3
    dataset = OneFlodDataSet(flod_dir, threshold)
    train_images, train_labels, train_scores = dataset.next_batch(128, [64, 64])
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
    vgg = vgg16(imgs, '/home/give/PycharmProjects/FaceDetection/fine_tuning_vgg16/vgg16_trained.npy', sess)
    train_features = sess.run(
        vgg.feature,
        feed_dict={
            vgg.imgs: train_images
        }
    )

    print np.shape(train_features)
    svr = MySVR(train_features, train_labels, train_scores, None, validation_labels, validation_scores)

    # ---start test-------
    count_number = 120
    test_imagess = Tools.split_arr(test_images, count_number)
    regionss = Tools.split_arr(regions, count_number)
    test_scoress = []
    for key in test_imagess.keys():
        test_images = test_imagess[key]
        regions = regionss[key]
        test_features = sess.run(
            vgg.feature,
            feed_dict={
                vgg.imgs: test_images
            }
        )
        test_scores = svr.do_predict(test_features)
        test_scoress.extend(test_scores)
        EQUAL_THRESHOLD = 0.5
        predicted_labels = []
        index = 0
        for score in test_scores:
            if score >= EQUAL_THRESHOLD:
                predicted_labels.append(1)
                # Tools.show_rect_in_image(
                #     test_image_path,
                #     [regions[index]],
                #     [1]
                # )
                Tools.save_rect_in_image(
                    test_image_path,
                    [regions[index]],
                    [1],
                    [score],
                    '/home/give/PycharmProjects/FaceDetection/fine_tuning_vgg16/result/trained'
                )
            else:
                predicted_labels.append(0)
            index += 1
        print 'labels shape is ', np.shape(predicted_labels)
        print 'regions shape is ', np.shape(regions)
        print 'face rect number is ', np.sum(predicted_labels)
    print 'test scores shapie ', np.shape(test_scoress)
    print 'regions copy shape is ', np.shape(regions_copy)
    test_scoress = np.array(test_scoress)
    regions_copy = np.array(regions_copy)
    EQUAL_THRESHOLD = 0.55
    index = [test_scoress >= EQUAL_THRESHOLD]
    test_scoress = test_scoress[index]
    regions_copy = regions_copy[index]
    Tools.non_maximum_suppression(test_image_path, test_scoress, regions_copy, type)

if __name__ == '__main__':
    detection_image('/home/give/Documents/dataset/FaceDetection/TenFolds/02/img_5.jpg', 'min')