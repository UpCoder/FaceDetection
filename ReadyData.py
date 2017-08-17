import numpy as np
import shutil
import os
ANNOTATION_DIR = '/home/give/Documents/dataset/FaceDetection/FDDB-folds'
IMAGE_DIR = '/home/give/Documents/dataset/FaceDetection/originalPics'
SAVE_PATH = '/home/give/Documents/dataset/FaceDetection/TenFolds'


# copy image from  originalPics to one folder
def to_one_folder(name_list_file_name):
    images_count = 0
    cur_id = os.path.basename(name_list_file_name.split('.')[0])[10:]
    cur_save_path = os.path.join(SAVE_PATH, cur_id)
    if not os.path.exists(cur_save_path):
        print '-'*15, 'making dir: ', cur_save_path, '-'*15
        os.mkdir(cur_save_path)
    print cur_id
    name_list_file = open(name_list_file_name)
    lines = name_list_file.readlines()
    for line in lines:
        image_path = os.path.join(IMAGE_DIR, line.replace('\n', '') + '.jpg')
        image_name = os.path.basename(image_path)
        if not os.path.exists(image_path):
            print 'not exists', image_path
            continue
        save_file_path = os.path.join(cur_save_path, image_name)
        shutil.copy(image_path, save_file_path)
        images_count += 1
        print 'copy finished ', save_file_path
    return images_count


def to_ten_folder():
    annotation_files = os.listdir(ANNOTATION_DIR)
    images_count = 0
    for annotation_file in annotation_files:
        if annotation_file.endswith('ellipseList.txt'):
            continue
        images_count += to_one_folder(os.path.join(ANNOTATION_DIR, annotation_file))
    print '-'*15, 'finish image transformation task. the image number is ', images_count, '-'*15

if __name__ == '__main__':
    to_ten_folder()
