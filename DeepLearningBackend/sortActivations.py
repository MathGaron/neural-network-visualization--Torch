from TensorflowBackend import TensorflowBackend
# from InputGenerators.LoaderBase import DataCaltech101Example

# test function
from scipy.misc import imread, imresize
import numpy as np

tfmodel = '/home-local/jizha16.extra.nobkp/data/ml/vgg16-tfmodel.meta'
# loader = DataCaltech101Example()

model = TensorflowBackend()
model.load_tf_model(tfmodel)


img = np.reshape(imresize(imread('../images/lena.png'), [224, 224]) / 255.0, [1, 224, 224, 3])
ims = img
model.forward(ims)


def sortActivations(model):
    activations = model.get_convolution_activation()
    # [[N, H, W, C], ]

    act_most = []  # [[N, C], ]
    for activation in activations:
        activation = np.asarray(activation)
        act_most.append(np.sum(activation, axis=(1, 2)))
    return act_most
