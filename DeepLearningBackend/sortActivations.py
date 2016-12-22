from DeepLearningBackend.TensorflowBackend import TensorflowBackend
# python3
# from InputGenerators.LoaderBase import DataCaltech101Example
# python2
import imp
InputGenerators = imp.load_source('LoaderBase', '../InputGenerators/LoaderBase.py')

import numpy as np

# select Torch or Tensorflow as backend
tfmodel = '/home-local/jizha16.extra.nobkp/data/ml/vgg16-tfmodel.meta'
model = TensorflowBackend()
model.load(tfmodel)


def sortActivations(activations):
    # activations: [[N, H, W, C], ...]
    most_activated = []  # [[N, C], ]
    for activation in activations:
        activation = np.asarray(activation)
        act_sum = np.sum(activation, axis=(0, 1, 2))  # sum or mean follow the same order
        ind = np.argsort(act_sum)[::-1]
        most_activated.append(ind)
    return most_activated  # just return an index


loader = InputGenerators.DataCaltech101Example()
loader.setClass('camera')
ims, labels = loader.next()
model.forward(ims)
activations = model.get_convolution_activation()
most_activated = sortActivations(activations)
print(most_activated)

# do a forward pass
loader = InputGenerators.DataCaltech101Example()
loader.setClass('pizza', 10)
# or only one
# ims = np.reshape(imresize(imread('../images/lena.png'), [224, 224]) / 255.0, [1, 224, 224, 3])
activations = []
for i in loader.length:
    ims, labels = loader.next()
    model.forward(ims)
    acts = model.get_convolution_activation()
    activations.append(acts)
    activations, most_activated = sortActivations(activations)
    # [[N, H, W, C], layer2...]], [top index of C, ]
