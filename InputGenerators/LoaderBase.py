import abc
import os
import glob2
from pprint import pprint
import cv2
import numpy as np
import math


class DataLoader(object):

    def __init__(self, data, labels=[], batch_size=100):
        self.data = data

        if labels is []:
            labels = range(len(data))
        self.labels = labels

        self.count = 0
        self.batch_size = batch_size
        self.length = math.floor(len(data) / self.batch_size) - 1

    def __iter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __enter__(self):
        return self

    def next(self):
        while self.count + self.batch_size <= len(self.data):
            self.count = self.count + self.batch_size
            batch_fname = self.data[self.count - self.batch_size: self.count]
            batch_label = self.labels[self.count - self.batch_size: self.count]
            return self.load_minibatch(batch_fname, batch_label)
        print('no more data')

    @abc.abstractmethod
    def load_minibatch(self, batch_fnames, batch_labels=[]):
        '''
        implement this part to return images
        '''


class DataCaltech101Example(DataLoader):

    def __init__(self):
        self.data_path = '/home-local/jizha16.extra.nobkp/data/ml/101_ObjectCategories'
        self.file_list = glob2.glob(os.path.join(self.data_path, '*/*.jpg'))
        self.labels = [os.path.split(os.path.split(p)[0])[1] for p in self.file_list]
        self.classes = list(set(self.labels))
        print(self.classes)

    def setClass(self, label='pizza', batch_size=100):
        indices = [i for i, s in enumerate(self.labels) if label in s]
        labels = [self.labels[i] for i in indices]
        file_list = [self.file_list[i] for i in indices]
        batch_size = min(batch_size, len(file_list))
        super(DataCaltech101Example, self).__init__(file_list, labels, batch_size)

    def load_minibatch(self, batch_fnames, batch_labels=[]):
        ims = []
        im_size = (224, 224)
        for fn in batch_fnames:
            ims.append(cv2.resize(cv2.imread(fn), im_size))
        return np.asarray(ims), batch_labels
        # return [N_batch, Height, Width, Channel]


if __name__ == '__main__':
    '''
    get dataset from:
    http://www.vision.caltech.edu/Image_Datasets/Caltech101/
    '''
    loader = DataCaltech101Example()
    loader.setClass(batch_size=10, label='pizza')
    pprint(loader.next()[0].shape)
    pprint('done')
