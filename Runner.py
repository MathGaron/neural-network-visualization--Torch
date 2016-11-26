import json
import os

import cv2
import numpy as np

from SpatialActivationViewer import SpatialActivationViewer
from InputGenerators.CameraInputGenerator import CameraInputGenerator
from DeepLearningBackend.TorchBackend import TorchBackend

activation_viewer = SpatialActivationViewer()
LAYER_SCREEN_SIZE = 800
screen_ratio = 0


def prepare_image(img, newsize):
    img = cv2.resize(img, (newsize, newsize), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    img = img.transpose((2, 0, 1))
    return img.reshape(1, 3, newsize, newsize)


def mouse_click(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        activation_viewer.filter_selection(x * screen_ratio, y * screen_ratio)

if __name__ == '__main__':

    model = TorchBackend()

    # load params
    path_to_config = "config/param.json"
    with open(path_to_config) as json_data:
        settings = json.load(json_data)

    # Default folder names
    filterResponsesFolderName = 'filter-responses'
    imageFolderName = 'images'

    # load/build model
    #flashlight.build_model()
    model_path = settings["caffe_model_path"]
    model.load_cafe_model(os.path.join(model_path, "VGG_CNN_M_deploy.prototxt"),
                          os.path.join(model_path, "VGG_CNN_M.caffemodel"))

    # Setup class name
    classes = []
    with open(settings["dataset_classe_file"]) as file:
        for line in file:
            classes.append(line)

    # run webcam
    input_generator = CameraInputGenerator()
    cv2.namedWindow("filters")
    cv2.setMouseCallback("filters", mouse_click)
    for frame in input_generator:
        # Capture and process image
        cv2.imshow("frame", frame)
        img = prepare_image(frame, 224)
        # output prediction
        output = model.predict(img)

        filters = model.get_convolution_activation()
        activation_viewer.update_filter_data(filters)
        filter_grid, filter_img = activation_viewer.draw(filters)
        screen_ratio = float(filter_grid.shape[0]) / float(LAYER_SCREEN_SIZE)
        cv2.imshow("filters", cv2.resize(filter_grid, (LAYER_SCREEN_SIZE, LAYER_SCREEN_SIZE), interpolation=cv2.INTER_CUBIC))
        print("layer mean activation : {}".format(activation_viewer.get_layer_mean_activation(filters)))
        if filter_img is not None:
            print("filter mean activation : {}".format(activation_viewer.get_selected_filter_mean_activation(filters)))
            filter_img = cv2.resize(filter_img, (300, 300), interpolation=cv2.INTER_CUBIC)
            cv2.imshow("filter_image", filter_img)

        # keyboard control
        k = cv2.waitKey(33)
        if k == 1113927:  # Esc key to stop
            break
        elif k == 1113939:  # left arrow
            activation_viewer.layer_selection_increment(1)
        elif k == 1113937:  # right arrow
            activation_viewer.layer_selection_increment(-1)
    cv2.destroyAllWindows()

