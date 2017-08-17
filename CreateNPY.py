import os
import numpy as np
from PIL import Image, ImageDraw
import Tools
from multiprocessing import Process, Pool
import gc
RESIZE_W = 224
RESIZE_H = 224
FLODS_DIR = '/home/give/Documents/dataset/FaceDetection/TenFolds'
NPY_DIR = '/home/give/Documents/dataset/FaceDetection/npy'
TEMP_DIR = '/home/give/Documents/dataset/FaceDetection/temp'
BOUDNING_BOX_SAVING_DIR = '/home/give/Documents/dataset/FaceDetection/Bounding_Box'
ANNOTATION_FILES_DIR = '/home/give/Documents/dataset/FaceDetection/FDDB-folds'
EQUAL_THRESHOLD = 0.3
SAVE_TEMP = True


# save selected bounding box image
def save_selected_image():
    create_single_npy('/home/give/Documents/dataset/FaceDetection/FDDB-folds/FDDB-fold-01-ellipseList.txt', save_selected=True)


# calculate the label(IoU value)
# image_path is the path of the calculated image
# selected_rects is the rectangles by using selective search method
# records is the annotation data
# save_flag is the flag that determined whether the selected resized image save
def calu_label(image_path, select_rects, records, save_flag=False):
    print 'calu label for ', image_path
    dir_name = os.path.dirname(image_path)
    id_name = os.path.basename(dir_name)
    image_name = os.path.basename(image_path).split('.')[0]
    images = []
    labels = []
    image = Tools.read_image(image_path)
    shape = list(np.shape(image))
    record_images_bg = []
    # get the annotataion image, it is convenient to calculation IoU
    for record in records:
        record_image_bg = np.ones([shape[0], shape[1]], dtype=np.uint8) * 150
        record_image_bg = Image.fromarray(record_image_bg)
        record_image_bg = Tools.ellipse_with_angle(
            record_image_bg,
            int(record[3]),
            int(record[4]),
            int(record[0] * 2),
            int(record[1] * 2),
            np.degrees(record[2]),
            color=(255, 255, 255),
        )
        record_image_bg = np.array(record_image_bg)
        record_images_bg.append(record_image_bg)
    count = 0
    one_count = 0
    for select_rect in select_rects:
        image_copy = image.copy()
        selected_image = np.array(image_copy)[
                         select_rect[1]:select_rect[1] + select_rect[3],
                         select_rect[0]:select_rect[0] + select_rect[2],
                         :]
        selected_image = Image.fromarray(np.asarray(selected_image, np.uint8))
        selected_image = selected_image.resize([RESIZE_W, RESIZE_H])
        select_image_bg = np.zeros([shape[0], shape[1]], dtype=np.uint8)
        select_image_bg = Image.fromarray(select_image_bg)
        select_image_bg_draw = ImageDraw.Draw(select_image_bg)
        select_image_bg_draw.rectangle(
            [select_rect[0], select_rect[1], select_rect[0] + select_rect[2], select_rect[1] + select_rect[3]],
            fill=255,
        )
        max_rate = 0
        select_image_bg = np.array(select_image_bg)
        for record_image_bg in record_images_bg:
            equal_number = np.sum((record_image_bg == 255) * (select_image_bg == 255))
            bing_number = np.sum(((record_image_bg == 255) + (select_image_bg == 255)) != 0)
            equal_rate = (equal_number * 1.0) / (bing_number * 1.0)     # euqal rate is IoU
            # if equal_number != 0:
            #     print equal_number, equal_rate
            max_rate = max(max_rate, equal_rate)
        count += 1
        if save_flag:
            save_path = os.path.join(BOUDNING_BOX_SAVING_DIR, id_name)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_path = os.path.join(save_path, image_name + '_' + str(max_rate) + '.jpg')
            selected_image.save(save_path)
        if max_rate >= EQUAL_THRESHOLD:
            images.append(np.array(selected_image))
            labels.append(max_rate)
            one_count += 1
        else:
            randomint = np.random.randint(100)
            if randomint > 80:
                images.append(np.array(selected_image))
                labels.append(max_rate)
        # labels.append(max_rate)
    # print 'one label count is ', one_count, 'selects rect number is ', len(select_rects)
    del record_images_bg
    del select_rects
    del image
    gc.collect()
    return images, labels


# create bounding box by annotation file content
# save the bounding box images after resizing matched size
def create_single_npy(annotation_file, save_selected = False):
    file = open(annotation_file)
    line = file.readline()
    base_name = os.path.basename(annotation_file)
    id = base_name[10:12]
    res_images = []
    res_labels = []
    res_count = []
    while line:
        # for every file
        image_path = line
        image_name = image_path[image_path.find('img'):-1]  # get image name
        image_path = os.path.join(FLODS_DIR, id, image_name+'.jpg')
        count = int(file.readline())    # get the number of annotation
        records = []    # len(records) = count
        for i in range(count):
            record = file.readline()
            record = record.replace('\n', '')
            record = record.replace('  ', ' ')
            record = record.split(' ')
            record = [(float(number)) for number in record]
            records.append(record)
        selected_rect_file = os.path.join(FLODS_DIR, id, image_name+'.txt')
        selected_rect_fileed = open(selected_rect_file)
        selected_rect_lines = selected_rect_fileed.readlines()
        selected_rects = []     # by using selective search method, we get the rectangle of bounding box
        for selected_rect_line in selected_rect_lines:
            selected_rect_line = selected_rect_line.replace(' \n', '')
            selected_rect_line = selected_rect_line.replace('\r\n', '')
            selected_rect_line = selected_rect_line[:-1]
            selected_rect = selected_rect_line.split(' ')
            # print selected_rect
            selected_rect = [int(number) for number in selected_rect]
            if selected_rect[2] == 0 or selected_rect[3] == 0:
                continue
            selected_rects.append(selected_rect)
        images, labels = calu_label(image_path, selected_rects, records, save_selected)
        res_images.extend(images)
        res_labels.extend(labels)
        res_count.append(len(images))
        line = file.readline()
    res_image_file = os.path.join(NPY_DIR, id+'_images.npy')
    res_labels_file = os.path.join(NPY_DIR, id+'_labels.npy')
    res_count_file = os.path.join(NPY_DIR, id+'_count.npy')
    np.save(
        res_image_file,
        res_images
    )
    np.save(
        res_labels_file,
        res_labels
    )
    np.save(
        res_count_file,
        res_count
    )


def create_npys(dir_path):
    ids_range = range(1, 8)
    print ids_range
    file_names = os.listdir(dir_path)
    for file_name in file_names:
        if not file_name.endswith('ellipseList.txt'):
            continue
        id = int(file_name[10:12])
        print id
        if not id in ids_range:
            continue
        file_path = os.path.join(dir_path, file_name)
        print '-'*15, file_path, '-'*15
        process = Process(
            target=create_single_npy,
            args=(
                file_path,
            )
        )
        process.start()


# return the annotation by dict
# dict[id][image_name][len(records)]
def create_annotation_dict(annotations_dir):
    files = os.listdir(annotations_dir)
    dict = {}
    for file in files:
        if not file.endswith('-ellipseList.txt'):
            continue
        id = int(file[10:12])
        cur_dict = {}
        file_path = os.path.join(annotations_dir, file)
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


def get_next_batch(batch_size, annotation_dict):
    res_images = []
    res_labels = []
    pool = Pool(processes=16)
    results = []
    for i in range(batch_size):
        id = "%02d" % np.random.randint(1, 8)
        cur_images_dir = os.path.join(FLODS_DIR, id)
        image_names = os.listdir(cur_images_dir)
        index = np.random.randint(len(image_names))
        image_name = image_names[index]
        while True:
            image_name = image_names[index]
            if image_name.endswith('.txt') or image_name.endswith('annotation.jpg') or image_name.endswith('rects.jpg') or image_name.endswith('diff.jpg'):
                index = np.random.randint(len(image_names))
                continue
            else:
                break
        image_name = image_name[:image_name.find('.')]
        image_path = os.path.join(cur_images_dir, image_name + '.jpg')
        records = annotation_dict[int(id)][image_name]
        selected = Tools.read_image_selected_rect(os.path.join(cur_images_dir, image_name + '.txt'))
        results.append(
            pool.apply_async(
                calu_label,
                [image_path, selected, records]
            )
        )
    pool.close()
    pool.join()
    for result in results:
        images, labels = result.get()
        res_images.extend(images)
        res_labels.extend(labels)
    # print np.shape(res_images)
    # print np.shape(res_labels)
    # print 'max IoU is ', np.max(res_labels)
    return res_images, res_labels

if __name__ == '__main__':
    # image_path = 'E:\\Resource\\DataSet\\FaceDetection\\TenFolds\\01\\img_17676.jpg'
    # records = [
    #     [37.099961, 29.000000, 1.433107, 28.453831, 37.664572,  1]
    # ]
    # draw_single_annotation(image_path, records)
    # im = Image.open(image_path)
    # im = ellipse_with_angle(im, x=28, y=37, major=37*2, minor=29*2, angle=82, color=(255, 0, 0, 255))
    # im = ellipse_with_angle(im, x=112, y=92, major=79*2, minor=49*2, angle=np.degrees(-1.457361), color=(255, 0, 255, 150))
    # # im = make_color_transparent(im,(0,0,0,255))
    # im.show()
    # create_single_npy('E:\\Resource\\DataSet\\FaceDetection\\FDDB-folds\\FDDB-fold-01-ellipseList.txt')
    # create_npys(ANNOTATION_FILES_DIR)
    # annotation_dict = create_annotation_dict(ANNOTATION_FILES_DIR)
    # print annotation_dict
    # get_next_batch(1, annotation_dict)
    save_selected_image()