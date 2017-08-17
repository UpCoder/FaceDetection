# -*- coding: utf-8 -*-
import selectivesearch
import numpy as np
from PIL import Image, ImageDraw
import os
from multiprocessing import Process
import Tools


def write_array_to_file(arr, file_path):
    file_obj = open(file_path, 'w')
    file_obj.writelines(arr)
    file_obj.close()
    print 'file save operation finised, ', file_path, ' the number of rects is ', len(arr)


def list_to_string(arr):
    res = ''
    for ele in arr:
        res += (str(ele) + ' ')
    res += '\n'
    return res


def draw_rect(im, rects):
    im_draw = ImageDraw.Draw(im)
    for rect in rects:
        im_draw.rectangle([rect[0], rect[1], rect[0]+rect[2], rect[1] + rect[3]], outline=(255, 0, 0))


def create_single_selected_rect(image_path, records=None):
    img = Tools.read_image(image_path)
    dir_name = os.path.dirname(image_path)
    base_name = os.path.basename(image_path).split('.')[0]
    rect_path = os.path.join(dir_name, base_name + '.txt')
    annotation_image_path = os.path.join(dir_name, base_name + '_annotation.jpg')
    rect_image_save_path = os.path.join(dir_name, base_name + '_rects.jpg')
    if os.path.exists(rect_image_save_path):
        print 'have exists ', rect_image_save_path
        return
    _, regions = selectivesearch.selective_search(
        np.array(img),
        scale=128,
        sigma=0.9,
        min_size=10
    )
    regions = [region['rect'] for region in regions]
    regions = list(set(t for t in regions))
    draw_rect(img, regions)

    img.save(rect_image_save_path)
    regions = [list_to_string(t) for t in regions]
    write_array_to_file(regions, rect_path)
    img = Tools.read_image(image_path)
    if records is None:
        return
    for record in records:
        img = Tools.ellipse_with_angle(
            img,
            record[3],
            record[4],
            record[0]*2,
            record[1]*2,
            np.degrees(record[2]),
            color=(255, 0, 0))
    img.save(annotation_image_path)


def processing_one_folder(fold_path, annotation_dict):
    image_names = os.listdir(fold_path)
    for image_name in image_names:
        if image_name.endswith('.jpg')\
                and not image_name.endswith('_rects.jpg')\
                and not image_name.endswith('_annotation.jpg')\
                and not image_name.endswith('_diff.jpg'):
            print 'will doing ', os.path.join(fold_path, image_name)
            create_single_selected_rect(os.path.join(fold_path, image_name), annotation_dict[image_name.split('.')[0]])


def create_selected_rects(fold_dir, annotations_dict):
    fold_names = os.listdir(fold_dir)
    for fold_name in fold_names:
        print '-'*15, 'processing the folder ', fold_name, '-'*15
        fold_path = os.path.join(fold_dir, fold_name)
        process = Process(
            target=processing_one_folder,
            args=(
                fold_path,
                annotations_dict[int(fold_name)],
            )
        )
        process.start()


def create_annotation_dict(annotations_dir):
    files = os.listdir(annotations_dir)
    dict = {}
    for file in files:
        if not file.endswith('-ellipseList.txt'):
            continue
        id = int(file[10:12])
        cur_dict = {}
        file_path = os.path.join(annotation_dir, file)
        opened_file = open(file_path)
        line = opened_file.readline()
        while line:
            image_name = line[line.find('img'):-1]
            count = int(opened_file.readline())
            records = []
            for i in range(count):
                record = opened_file.readline()
                record = record.replace('\n', '')
                record = record.replace('  ', ' ')
                record = record.split(' ')
                record = [int(float(number)) for number in record]
                records.append(record)
            cur_dict[image_name] = records
            line = opened_file.readline()
        dict[id] = cur_dict
    return dict

if __name__ == '__main__':
    annotation_dir = 'E:\Resource\DataSet\FaceDetection\FDDB-folds'
    annotation_dict = create_annotation_dict('E:\\Resource\\DataSet\\FaceDetection\\FDDB-folds')
    print annotation_dict
    print annotation_dict[1]['img_1027']
    fold_dir = 'E:\\Resource\\DataSet\\FaceDetection\\TenFolds'
    create_selected_rects(fold_dir, annotation_dict)
    # test_image_path = 'E:\\Resource\\DataSet\\FaceDetection\\TenFolds\\01\\img_1027.jpg'
    # create_single_selected_rect(test_image_path)
    # print 'ok'