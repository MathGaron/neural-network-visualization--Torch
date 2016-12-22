from InputGenerators.LoaderBase import DataLoader
import os
import glob2
import math
import cv2


class Caltech101Dataset(DataLoader):
    def __init__(self, path, label):
        self.data_path = path
        self.file_list = glob2.glob(os.path.join(self.data_path, '*/*.jpg'))
        self.labels = [os.path.split(os.path.split(p)[0])[1] for p in self.file_list]
        self.classes = list(set(self.labels))
        self.setClass_(label)

    def setClass_(self, label):
        indices = [i for i, s in enumerate(self.labels) if label in s]
        labels = [self.labels[i] for i in indices]
        file_list = [self.file_list[i] for i in indices]

        self.data = file_list

        if labels is []:
            labels = range(len(data))
        self.labels = labels

        self.count = 0
        self.length = math.floor(len(self.data)) - 1

    def stop_iteration(self):
        return self.count + 1 <= len(self.data)

    def load_minibatch(self):
        self.count += 1
        batch_fname = self.data[self.count - 1: self.count]
        return cv2.imread(batch_fname[0])


if __name__ == '__main__':
    '''
    get dataset from:
    http://www.vision.caltech.edu/Image_Datasets/Caltech101/
    '''
    loader = Caltech101Dataset('/home/mathieu/Dataset/101_ObjectCategories')
    loader.setClass(label='beaver')
    for data in loader:
        cv2.imshow("test", data)
        cv2.waitKey()
    print('done')
