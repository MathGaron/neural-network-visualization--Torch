
import PyTorchHelpers
import cv2
import os
import numpy as np
import math


def torch2numpy(data):
    if isinstance(data, dict):
        for key, value in data.iteritems():
            data[key] = value.asNumpyTensor()
    return data


def dict2list(dict):
    index = [k for k in dict.keys()]
    index.sort()
    return [dict[i] for i in index]


def draw_2d_filters(filters):
    filters = np.squeeze(filters)
    n, w, h = filters.shape
    mosaic_size = math.ceil(math.sqrt(n))
    mosaic = np.zeros((mosaic_size * w, mosaic_size * h), dtype=np.uint8)
    for i, individual_filter in enumerate(filters):
        x = int(i % mosaic_size) * h
        y = int(i / mosaic_size) * w
        mosaic[x:x + h, y:y + w] = cv2.convertScaleAbs(individual_filter)
    return mosaic

if __name__ == '__main__':

    Flashlight = PyTorchHelpers.load_lua_class("torch-nn-viz-example.lua", 'Flashlight')
    flashlight = Flashlight("cpu")

    # Default folder names
    filterResponsesFolderName = 'filter-responses'
    imageFolderName = 'images'

    flashlight.clear_gnu_plots()

    filename = os.path.join(imageFolderName, 'lena.png')
    newsize = 32
    img = cv2.imread(filename)
    img = cv2.resize(img, (newsize, newsize), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    img = img.reshape(1, 3, newsize, newsize)

    flashlight.build_model()
    filters = flashlight.get_layer_responses(img)
    filters = torch2numpy(filters)
    filters = dict2list(filters)
    for filter in filters:
        mosaic = draw_2d_filters(filter)
        mosaic = cv2.resize(mosaic, (800, 800), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("test", mosaic)
        cv2.waitKey()

