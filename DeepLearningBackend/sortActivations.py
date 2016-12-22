import numpy as np
from scipy.misc import imresize

def sortActivations(activations):
    # activations: [[N, Channels, Width, Height], ...]
    most_activated = []  # [[N, C], ...]ind = np.argsort(act_sum)[::-1]
    for activation in activations:
        activation = np.asarray(activation)
        act_sum = np.sum(activation, axis=(0, 2, 3))  # sum or mean follow the same order
        ind = np.argsort(act_sum)[::-1]
        most_activated.append(ind)
    return most_activated  # just return an index


if __name__ == '__main__':
    '''
    Call this function to test
    '''
    from TensorflowBackend import TensorflowBackend
    from Caltech101Dataset import Caltech101Dataset
    # select Torch or Tensorflow as backend
    tfmodel = '/home-local/jizha16.extra.nobkp/data/ml/vgg16-tfmodel.meta'
    dataset_path = '/home-local/jizha16.extra.nobkp/data/ml/101_ObjectCategories'
    # choose a Backend model
    model = TensorflowBackend()
    model.load(tfmodel)

    ''' do the following in the main line '''
    # get the data
    loader = Caltech101Dataset(dataset_path, '')
    loader.setClass_('camera')
    batch = []
    for data in loader:
        data = imresize(data, [224, 224, 1])
        batch.append(data)
    batch = np.asarray(batch)  # [N, H, W, C])
    print('done')

    # do a forward pass
    model.forward(batch)

    convos, linear = model.get_activation()
    most_activated_convos = sortActivations(convos)
    most_activated_linear = sortActivations(linear)

    print(most_activated_convos[0].shape)
    print(most_activated_linear[0].shape)
