from PIL import ImageDraw, Image
import numpy as np
import selectivesearch
import os


def ellipse_with_angle(im, x, y, major, minor, angle, color):
    # take an existing image and plot an ellipse centered at (x,y) with a
    # defined angle of rotation and major and minor axes.
    # center the image so that (x,y) is at the center of the ellipse
    x -= int(major/2)
    y -= int(major/2)

    # create a new image in which to draw the ellipse
    im_ellipse = Image.new('RGBA', (major,major), (255,255,255,0))
    draw_ellipse = ImageDraw.Draw(im_ellipse, "RGBA")

    # draw the ellipse
    ellipse_box = (0,int(major/2-minor/2),major,int(major/2-minor/2)+minor)
    draw_ellipse.ellipse(ellipse_box, fill=color)

    # rotate the new image
    rotated = im_ellipse.rotate(angle)
    rx, ry = rotated.size

    # paste it into the existing image and return the result
    im.paste(rotated, (x, y, x+rx, y+ry), mask=rotated)
    return im


def read_image(image_path):
    im = Image.open(image_path)
    im = np.array(im)
    shape = list(np.shape(im))
    if len(shape) == 2:
        shape.append(3)
        res_im = np.zeros(shape)
        for i in range(3):
            res_im[:, :, i] = im
    elif len(shape) == 3:
        res_im = im
    return Image.fromarray(np.asarray(res_im, np.uint8))


def read_image_selected_rect(path):
    selected_rect_fileed = open(path)
    selected_rect_lines = selected_rect_fileed.readlines()
    selected_rects = []
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
    return selected_rects


def conver_score2label(scores, threshold):
    labels = []
    for score in scores:
        if score >= threshold:
            labels.append([1, 0])
        else:
            labels.append([0, 1])
    return labels


def create_image_selected_rectangle(image_path):
    image = read_image(image_path)
    shape = list(np.shape(image))
    if shape[1] * shape[0] > 1024*1024:
        image = image.resize([1024, 1024])
    _, regions = selectivesearch.selective_search(
        np.array(image),
        scale=256,
        sigma=0.9,
        min_size=10
    )
    regions = [region['rect'] for region in regions]
    regions = list(set(t for t in regions))
    regions_new = []
    for region in regions:
        if region[2] == 0 or region[3] == 0:
            continue
        regions_new.append(region)
    regions = regions_new
    res = []
    image.show()
    for region in regions:
        image_copy = image.copy()
        selected_image = np.array(image_copy)[
                         region[1]:region[1] + region[3],
                         region[0]:region[0] + region[2],
                         :]
        shape = list(np.shape(selected_image))
        selected_image = Image.fromarray(np.asarray(selected_image, np.uint8))
        selected_image = selected_image.resize([224, 224])
        res.append(np.array(selected_image))
    return np.array(res), regions


def show_rect_in_image(image_path, rects, labels):
    image = read_image(image_path)
    image_draw = ImageDraw.Draw(image)
    for index, rect in enumerate(rects):
        if labels[index] == 1:
            image_draw.rectangle(
                [rect[0], rect[1], rect[0]+rect[2], rect[1] + rect[3]],
                outline=(255, 0, 0)
            )
    image.show()


def save_rect_in_image(image_path, rects, labels, scores, save_dir):
    image = read_image(image_path)
    image_name = os.path.basename(image_path).split('.')[0]
    for index, rect in enumerate(rects):
        if labels[index] == 1:
            image_copy = image.copy()
            selected_image = np.array(image_copy)[
                             rect[1]:rect[1] + rect[3],
                             rect[0]:rect[0] + rect[2],
                             :]
            selected_image = Image.fromarray(np.asarray(selected_image, np.uint8))
            selected_image = selected_image.resize([224, 224])
            save_path = os.path.join(save_dir, image_name + '_' + str(scores[index]) + '.jpg')
            selected_image.save(save_path)


def split_arr(arr, count):
    res = {}
    index = 0
    count_index = 0
    while index < len(arr):
        cur_arr = []
        for i in range(count):
            if index >= len(arr):
                break
            cur_arr.append(arr[index])
            index += 1
        res[count_index] = cur_arr
        count_index += 1
    return res

if __name__ == '__main__':
    image_path = 'E:\\Resource\\DataSet\\FaceDetection\\TenFolds\\01\\img_591.jpg'
    record = [123, 85, 1.265839, 269, 161, 1]
    image = Image.open(image_path)
    image = ellipse_with_angle(
        image,
        record[3],
        record[4],
        record[0] * 2,
        record[1] * 2,
        np.degrees(record[2]),
        color=(255, 0, 0)
    )
    image.show()