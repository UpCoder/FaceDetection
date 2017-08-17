import os
import numpy as np
import Tools


class OneFlodDataSet:
    def __init__(self, flod_dir, label_threshold):
        self.flod_dir = flod_dir
        self.label_threshold = label_threshold
        self.bgs, self.faces, self.bg_scores, self.face_scores = self.mark_label()
        print 'bg number is ', len(self.bgs)
        print 'faces number is ', len(self.faces)

    # return every image mark label
    # if it less than label_threshold
    # if it more than and equal label_threshold
    def mark_label(self):
        files = os.listdir(self.flod_dir)
        backgournds = []
        bg_scores = []
        faces = []
        face_scores = []
        for file_name in files:
            IoU = file_name[file_name.find('_', 4)+1:file_name.find('.jpg')]
            if float(IoU) < self.label_threshold:
                backgournds.append(file_name)
                bg_scores.append(float(IoU))
            else:
                faces.append(file_name)
                face_scores.append(float(IoU))
        return backgournds, faces, bg_scores, face_scores

    # create one batch data
    # batch_size is the numebr of batch
    # distribution is the array
    #   distribution[0] is the number of negative sample(background)
    #   distribution[1] is the number of positive sample(face)
    def next_batch(self, batch_size, distribution):
        batch_data = []
        batch_labels = []
        batch_scores = []
        random_indexs = range(batch_size)
        np.random.shuffle(random_indexs)
        for i in range(distribution[0]):
            batch_labels.append([1, 0])
            random_index = np.random.randint(len(self.bgs))
            batch_scores.append(self.bg_scores[random_index])
            batch_data.append(np.array(Tools.read_image(os.path.join(self.flod_dir, self.bgs[random_index]))))
        for i in range(distribution[1]):
            batch_labels.append([0, 1])
            random_index = np.random.randint(len(self.faces))
            batch_scores.append(self.face_scores[random_index])
            batch_data.append(np.array(Tools.read_image(os.path.join(self.flod_dir, self.faces[random_index]))))
        batch_data = np.array(batch_data)
        batch_labels = np.array(batch_labels)
        batch_scores = np.array(batch_scores)
        batch_data = batch_data[random_indexs]
        batch_labels = batch_labels[random_indexs]
        batch_scores = batch_scores[random_indexs]
        return batch_data, batch_labels, batch_scores
if __name__ == '__main__':
    flod_dir = '/home/give/Documents/dataset/FaceDetection/Bounding_Box/01'
    threshold = 0.3
    dataset = OneFlodDataSet(flod_dir, threshold)
    data, label, socres = dataset.next_batch(128, [32, 96])
    print label
    print np.shape(data), np.shape(label)
