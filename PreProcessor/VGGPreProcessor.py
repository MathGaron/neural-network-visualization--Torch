import cv2
import numpy as np


class VGGPreProcessor:
    def __init__(self, input_size):
        self.input_size = input_size

    def preprocess_input(self, input):
        img = cv2.resize(input, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
        img = img.transpose((2, 0, 1))
        return img.reshape(1, 3, self.input_size, self.input_size)

    @staticmethod
    def show_input(input):
        cv2.imshow("VGG Input", input)

    def check_input(self, input):
        """
        This function should make sure that the input is ok, else throw an error
        :param input:
        :return:
        """
        pass