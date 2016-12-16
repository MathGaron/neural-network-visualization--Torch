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
        self.batch_size = 10
        self.length = math.floor(len(data) / self.batch_size) - 1

    def __iter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __enter__(self):
        return self

    def next(self):
        while self.count + self.batch_size < len(self.data):
            self.count = self.count + self.batch_size
            batch_fname = self.data[self.count - self.batch_size: self.count]
            batch_label = self.labels[self.count - self.batch_size: self.count]
            return self.load_minibatch(batch_fname, batch_label)

    @abc.abstractmethod
    def load_minibatch(self, batch_fnames, batch_labels=[]):
        '''
        implement this part to return images
        '''


class DataCaltech101Example(DataLoader):

    def __init__(self, batch_size=100):
        data_path = '/home-local/jizha16.extra.nobkp/data/ml/101_ObjectCategories'
        file_list = glob2.glob(os.path.join(data_path, '*/*.jpg'))
        labels = [os.path.split(os.path.split(p)[0])[1] for p in file_list]

        super(DataCaltech101Example, self).__init__(file_list, labels, batch_size)

    def load_minibatch(self, batch_fnames, batch_labels=[]):
        ims = []
        im_size = (224, 224)
        for fn in batch_fnames:
            ims.append(cv2.resize(cv2.imread(fn), im_size))
        return np.asarray(ims)
        # return [N_batch, Height, Width, Channel]


if __name__ == '__main__':
    '''
    get dataset from:
    http://www.vision.caltech.edu/Image_Datasets/Caltech101/
    '''
    loader = DataCaltech101Example(batch_size=100)
    pprint(loader.next().shape)
    pprint('done')
