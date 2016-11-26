
import PyTorchHelpers
import cv2
import os
import numpy as np
import json

from SpatialActivationViewer import SpatialActivationViewer

from sklearn import datasets

activation_viewer = SpatialActivationViewer()
LAYER_SCREEN_SIZE = 800
screen_ratio = 0

def torch2numpy(data):
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = value.asNumpyTensor()
    return data


def dict2list(dict):
    index = [k for k in dict.keys()]
    index.sort()
    return [dict[i] for i in index]


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

    # load/build model
    #flashlight.build_model()
    model_path = settings["caffe_model_path"]
    flashlight.load_caffe_model(os.path.join(model_path, "VGG_CNN_M_deploy.prototxt"),
                                os.path.join(model_path, "VGG_CNN_M.caffemodel"))
    # Setup class name
    classes = []
    with open(settings["dataset_classe_file"]) as file:
        for line in file:
            classes.append(line)

    # run webcam
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("filters")
    cv2.setMouseCallback("filters", mouse_click)
    filter_selection = 0
    while (True):
        # Capture and process image
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        img = prepare_image(frame, 224)
        # output prediction
        output = flashlight.predict(img).asNumpyTensor()
        #print classes[np.argmax(output)]
        filters = flashlight.get_convolution_activation()
        filters = torch2numpy(filters)
        filters = dict2list(filters)
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

        #else:
        #    print(k)
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

