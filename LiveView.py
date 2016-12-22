import json
import os
import sys
import cv2
import time
import numpy as np

import ImageOptimization
from ImageGenerator import deep_dream_optimize
from SpatialActivationViewer import SpatialActivationViewer
from InputGenerators.CameraInputGenerator import CameraInputGenerator
from InputGenerators.Caltech101Dataset import Caltech101Dataset
from DeepLearningBackend.TorchBackend import TorchBackend
from PreProcessor.VGGPreProcessor import VGGPreProcessor

activation_viewer = SpatialActivationViewer()
LAYER_SCREEN_SIZE = 800
screen_ratio = 0


def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        activation_viewer.filter_selection(x * screen_ratio, y * screen_ratio)


def backprop_layer(image, model):
    image = preprocessor.preprocess_input(image)
    out = model.forward(image)
    grads = model.backward_layer(2)
    image = preprocessor.preprocess_inverse(image)
    cv2.imshow("test", image)
    cv2.waitKey()


def get_activation_sum(activations):
    most_activated = []
    for activation in activations:
        activation = np.asarray(activation)
        act_sum = np.sum(activation, axis=(0, 2, 3))  # sum or mean follow the same order
        most_activated.append(act_sum)
    return most_activated


def sort_indexes(activations):
    indexes = []
    for activation in activations:
        ind = np.argsort(activation)[::-1]
        indexes.append(ind)
    return indexes

if __name__ == '__main__':

    # load params
    path_to_config = "config/param.json"
    with open(path_to_config) as json_data:
        settings = json.load(json_data)

    # Default folder names
    filterResponsesFolderName = 'filter-responses'
    imageFolderName = 'images'

    # load/build model
    if settings["deep_learning_backend"] == "torch":
        model = TorchBackend(settings["processing_backend"])
        model_path = settings["model_path"]
        model.load_cafe_model(os.path.join(model_path, "VGG_CNN_F_deploy.prototxt"),
                              os.path.join(model_path, "VGG_CNN_F.caffemodel"))
    else:
        print("The deep learning backend : {} is not recognized... exiting".format(settings["deep_learning_backend"]))
        sys.exit(-1)

    # Setup preprocessor and optimizers
    preprocessor = VGGPreProcessor(224)

    # Setup class name
    classes = []
    with open(settings["dataset_classe_file"]) as file:
        for line in file:
            classes.append(line)

    # run webcam or dataset
    #input_generator = CameraInputGenerator()
    input_generator = Caltech101Dataset(settings["data_path"], 'pizza')

    cv2.namedWindow("filters")
    cv2.setMouseCallback("filters", mouse_click)
    filters = None
    fps = time.time()
    convo_accumulator = []
    for input in input_generator:
        VGGPreProcessor.show_input(input)

        processed_input = preprocessor.preprocess_input(input)
        model.forward(processed_input)

        convo_filters, linear_filters = model.get_activation()
        sorted_convo = get_activation_sum(convo_filters)
        if convo_accumulator:
            for i in range(len(sorted_convo)):
                convo_accumulator[i] += sorted_convo[i]
        else:
            for values in sorted_convo:
                convo_accumulator.append(values)

        while time.time() - fps <= float(settings["fps"]):

            activation_viewer.update_filter_data(convo_filters)
            filter_grid, filter_img = activation_viewer.draw(convo_filters)
            screen_ratio = float(filter_grid.shape[0]) / float(LAYER_SCREEN_SIZE)
            cv2.imshow("filters",
                       cv2.resize(filter_grid, (LAYER_SCREEN_SIZE, LAYER_SCREEN_SIZE), interpolation=cv2.INTER_CUBIC))

            if filter_img is not None:
                mean, index = activation_viewer.get_selected_filter_mean_activation(convo_filters)
                filter_img = cv2.resize(filter_img, (300, 300), interpolation=cv2.INTER_CUBIC)
                cv2.imshow("filter #{}".format(index), filter_img)

            # keyboard control
            k = cv2.waitKey(30)
            if k == 1048691:
                indexes = sort_indexes(convo_accumulator)
                # deep dream example:
                b, g, r = VGGPreProcessor.getMeans()
                dream_optimizer = ImageOptimization.Optimizer(model)
                random_image = dream_optimizer.generate_gaussian_image((224, 224, 3), r, g, b)
                dream = deep_dream_optimize(dream_optimizer, preprocessor, random_image,
                                            iterations=70,
                                            octave_n=4,
                                            octave_scale=1.7,
                                            imagenet_index=583,
                                            gradient_energy=0.1,
                                            debug_view=True)
                stretch = cv2.resize(dream, (224 * 3, 224 * 3), interpolation=cv2.INTER_CUBIC)
                cv2.imshow("Dream", stretch)
                cv2.imwrite("output.png", stretch)
                cv2.waitKey()
            if k == 1048603:  # Esc key to stop
                sys.exit(0)
            if k == 1048608:
                break
            elif k == 1113937:  # left arrow
                activation_viewer.layer_selection_increment(1)
                print("layer selected : {}".format(activation_viewer.layer_selected))
            elif k == 1113939:  # right arrow
                activation_viewer.layer_selection_increment(-1)
                print("layer selected : {}".format(activation_viewer.layer_selected))
        fps = time.time()
    cv2.destroyAllWindows()
