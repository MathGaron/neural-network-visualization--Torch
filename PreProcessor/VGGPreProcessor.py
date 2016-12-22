import cv2
import numpy as np


class VGGPreProcessor:
    def __init__(self, input_size):
        self.input_size = input_size

    def preprocess_input(self, input):
        img = cv2.resize(input, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        b, g, r = VGGPreProcessor.getMeans()
        img[:, :, 2] -= b
        img[:, :, 1] -= g
        img[:, :, 0] -= r
        img = img.transpose((2, 0, 1))
        return img[np.newaxis, :, :, :]

    def preprocess_inverse(self, input):
        ret = input.copy()
        ret = ret[0, :, :, :]
        ret = ret.transpose((1, 2, 0))
        b, g, r = VGGPreProcessor.getMeans()
        ret[:, :, 2] += b
        ret[:, :, 1] += g
        ret[:, :, 0] += r
        return ret

    @staticmethod
    def show_input(input):
        cv2.imshow("VGG Input", input)

    @staticmethod
    def getMeans():
        blue = 103.939
        green = 116.779
        red = 123.68
        return blue, green, red

    def check_input(self, input):
        """
        This function should make sure that the input is ok, else throw an error
        :param input:
        :return:
        """
        pass