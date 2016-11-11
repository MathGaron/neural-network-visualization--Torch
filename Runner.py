
import PyTorchHelpers
import cv2
import os
import numpy as np

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
    flashlight.get_layer_responses(img)
    flashlight.visualize_filter_responses(img, 'frog-1', True, filterResponsesFolderName)
