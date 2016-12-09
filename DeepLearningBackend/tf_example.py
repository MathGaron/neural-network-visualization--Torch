import numpy as np
from scipy.misc import imread, imresize
import tensorflow as tf
import cv2
from pprint import pprint
import time
from SpatialActivationViewer import SpatialActivationViewer


class GraphViewerTF:

    def __init__(self, sess):
        self.graph = ''
        self.n_layers = 0
        self. sess = sess

        # vgg example,
        # todo @abc
        images = tf.placeholder("float", [None, 224, 224, 3])

    def read_graph(self, fname=''):
        print 'Loading graph from %s' % (fname)
        start_time = time.time()
        f = open(fname, mode='rb')
        fileContent = f.read()

        # vgg example, one input
        # todo @abc
        self.inputs = tf.placeholder("float", [None, 224, 224, 3])

        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fileContent)
        tf.import_graph_def(graph_def, input_map={"images": self.inputs})
        graph = tf.get_default_graph()
        self.graph = graph

        print "graph loaded from disk"
        print("Time : {}".format(time.time() - start_time))
        init = tf.initialize_all_variables()
        self.sess.run(init)

    def predict(self, img):
        return self.forward_pass(img)

    def forward_pass(self, img):
        '''
            run once for every image
        '''
        print 'prediction:'
        self.get_tensor_names()
        tensors = self.tensors['Relu']
        feed_dict = {self.inputs: img}
        # todo: this is super slow!!!
        outputs = self.sess.run(tensors, feed_dict=feed_dict)
        ot = [np.reshape(outputs[0], [64, 224, 224])]
        activation_viewer.update_filter_data(ot)

        filter_grid, filter_img = activation_viewer.draw(ot)
        print 'predict done'
        '''
        *
        *
        list of tensors could be
        *
        assert(type(*some output *).__module__ == 'numpy')  # not a list
        * there may be some way to direct get the layer, if so do:
        https:
            https://github.com/kvfrans/feature-visualization/blob/master/main.py
        '''

    def get_tensor_names(self, newtype=''):
        # update self.tensors dictionary
        '''
        return the tensor nodes, get any tensor by calling:
            self.tensors['type of the tensor, like Relu, Conv2D']
        as default, return all the Conv2D and Relu tensors
        newtype='sigmoid', will also return the sigmoid tensors
        '''
        self.tensors = {'Conv2D': None, 'Relu': None}
        self._update_tensors(tensor_type='Conv2D')
        self._update_tensors(tensor_type='Relu')
        if newtype != '':
            self.update_tensors(tensor_type=newtype)
        return self.tensors

    def _update_tensors(self, tensor_type='Conv2D'):
        '''
        Add tensor to self.tensors
        '''
        graph = self.graph
        layers = [op.name for op in graph.get_operations() if op.type == tensor_type and 'import/' in op.name]
        tensors = [graph.get_tensor_by_name(name + ':0') for name in layers]
        self.tensors.update({tensor_type: tensors})

    def get_convolution_activation(self, tensor_name):
        '''
        have to know the name of the tensor
        '''

    def helper_list_tensor(self):
        pass

if __name__ == '__main__':
    activation_viewer = SpatialActivationViewer()
    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:  # todo, keep sess alive
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    graphTF = GraphViewerTF(sess)
    fname = "/home-local/jizha16.extra.nobkp/data/deeplearning/vgg16.tfmodel"
    graphTF.read_graph(fname)
    img = imresize(imread('images/lena.png'), [224, 224]) / 255.0
    img = img.reshape((1, 224, 224, 3))
    start_time = time.time()
    graphTF.forward_pass(img)
    print("Time : {}".format(time.time() - start_time))
    print 'Close'
    sess.close()
    # with graphTF.sess:
