import json
import os
import sys
import cv2
import numpy as np
import scipy.ndimage as nd

from DeepLearningBackend.TorchBackend import TorchBackend
from PreProcessor.VGGPreProcessor import VGGPreProcessor
import ImageOptimization


# Exemple code to run deep dream activation
def deep_dream_optimize(optimizer, preprocessor, image, iterations=10):
    # we could show the filters at each iterations here...
    octave_n = 4
    octave_scale = 1.7
    octaves = [image.astype(np.float32)]
    for i in range(octave_n - 1):
        octaves.append(nd.zoom(octaves[-1], (1.0 / octave_scale, 1.0 / octave_scale, 1), order=1).astype(np.float32))
    detail = np.zeros_like(octaves[-1])  # allocate image for network-produced details
    for j, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[:2]
        if j > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[:2]
            detail = nd.zoom(detail, (1.0 * w / w1, 1.0 * h / h1, 1), order=1)
            h1, w1 = octave_base.shape[:2]
            detail = cv2.resize(detail, (w1, h1))
        image = octave_base + detail
        image = preprocessor.preprocess_input(image)
        for i in range(iterations):
            image = optimizer.make_step(image, ImageOptimization.Optimizer.objective_maximize_class, i, 1.7, energy=0.1,
                                        index=63)  # 76:tarantula   #130:flamingo  #113:snail #340:zebra #323:monarch 327:seastar  980:volcano
            cv2.imshow("test", preprocessor.preprocess_inverse(image).astype(np.uint8))
            cv2.waitKey(20)
        image = preprocessor.preprocess_inverse(image)
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
        detail = image - octave_base
    return image.astype(np.uint8)


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
        model_path = settings["caffe_model_path"]
        model.load_cafe_model(os.path.join(model_path, "VGG_CNN_F_deploy.prototxt"),
                              os.path.join(model_path, "VGG_CNN_F.caffemodel"))
    else:
        print("The deep learning backend : {} is not recognized... exiting".format(settings["deep_learning_backend"]))
        sys.exit(-1)

    # Setup preprocessor and optimizers
    preprocessor = VGGPreProcessor(224)
    dream_optimizer = ImageOptimization.Optimizer(model)

    # deep dream example:
    b, g, r = VGGPreProcessor.getMeans()
    random_image = dream_optimizer.generate_gaussian_image((224, 224, 3), r, g, b)
    dream = deep_dream_optimize(dream_optimizer, preprocessor, random_image, iterations=100)
    cv2.imshow("Dream", cv2.resize(dream, (224 * 3, 224 * 3), interpolation=cv2.INTER_CUBIC))
    cv2.waitKey()
