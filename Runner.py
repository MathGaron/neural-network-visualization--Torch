import json
import os

import cv2

from SpatialActivationViewer import SpatialActivationViewer
from InputGenerators.CameraInputGenerator import CameraInputGenerator
from DeepLearningBackend.TorchBackend import TorchBackend
from PreProcessor.VGGPreProcessor import VGGPreProcessor

activation_viewer = SpatialActivationViewer()
LAYER_SCREEN_SIZE = 800
screen_ratio = 0


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

    preprocessor = VGGPreProcessor(224)

    # Setup class name
    classes = []
    with open(settings["dataset_classe_file"]) as file:
        for line in file:
            classes.append(line)

    # run webcam
    input_generator = CameraInputGenerator()
    cv2.namedWindow("filters")
    cv2.setMouseCallback("filters", mouse_click)
    filters = None
    for input in input_generator:
        # Capture and process image
        VGGPreProcessor.show_input(input)
        input = preprocessor.preprocess_input(input)
        # output prediction
        output = model.predict(input)

        model.get_convolution_filters()

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

