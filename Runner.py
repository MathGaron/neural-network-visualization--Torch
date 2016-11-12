
import PyTorchHelpers
import cv2
import os
import numpy as np
import math
import json

from sklearn import datasets


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


def prepare_image(img, newsize):
    img = cv2.resize(img, (newsize, newsize), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    img = img.transpose((2, 0, 1))
    return img.reshape(1, 3, newsize, newsize)


if __name__ == '__main__':
    # load lua files
    Flashlight = PyTorchHelpers.load_lua_class("torch-nn-viz-example.lua", 'Flashlight')
    flashlight = Flashlight("cuda")

    # load params
    path_to_config = "config/param.json"
    with open(path_to_config) as json_data:
        settings = json.load(json_data)

    # Default folder names
    filterResponsesFolderName = 'filter-responses'
    imageFolderName = 'images'

    flashlight.clear_gnu_plots()


    #flashlight.build_model()
    model_path = settings["caffe_model_path"].encode('ascii', 'ignore')
    flashlight.load_caffe_model(os.path.join(model_path, "VGG_CNN_M_deploy.prototxt"),
                                os.path.join(model_path, "VGG_CNN_M.caffemodel"))
    classes = []
    with open(settings["dataset_classe_file"]) as file:
        for line in file:
            classes.append(line)

    cap = cv2.VideoCapture(0)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        cv2.imshow('frame', frame)
        img = prepare_image(frame, 224)
        output = flashlight.predict(img).asNumpyTensor()
        print classes[np.argmax(output)]

        # Display the resulting frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

"""
    filters = flashlight.get_layer_responses(img)
    filters = torch2numpy(filters)
    filters = dict2list(filters)
    for filter in filters:
        mosaic = draw_2d_filters(filter)
        mosaic = cv2.resize(mosaic, (800, 800), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("filters", mosaic)
        cv2.waitKey()
"""

