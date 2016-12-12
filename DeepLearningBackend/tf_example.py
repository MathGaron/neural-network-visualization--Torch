import numpy as np
from scipy.misc import imread, imresize
import tensorflow as tf
import cv2
from pprint import pprint
import time
# from SpatialActivationViewer import SpatialActivationViewer


class GraphViewerTF:

    def __init__(self, sess):
        self.graph = ''
        self.n_layers = 0
        self. sess = sess

        # vgg example,
        # todo @abc
        images = tf.placeholder("float", [None, 224, 224, 3])

    def read_graph(self, fname=''):
        print('Loading graph from {}'.format(fname))
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

        print("graph loaded from disk")
        print("Time : {}".format(time.time() - start_time))
        init = tf.initialize_all_variables()
        self.sess.run(init)

    def predict(self, img):
        return self.forward_pass(img)

    def forward_pass(self, img):
        '''
            run once for every image
        '''
        print('prediction:')
        self.get_tensor_names()
        tensors = self.tensors['Relu']
        feed_dict = {self.inputs: img}
        # todo: this is super slow!!!
        start_time = time.time()
        outputs = self.sess.run(tensors[0], feed_dict=feed_dict)
        print("predict done, 1 opt. Time : %.3f" % (time.time() - start_time))
        start_time = time.time()
        outputs = self.sess.run(tensors, feed_dict=feed_dict)
        ot = [np.reshape(outputs[0], [64, 224, 224])]
        # activation_viewer.update_filter_data(ot)

        # filter_grid, filter_img = activation_viewer.draw(ot)
        print("predict done, %d opts. Time %.3f:" % (len(tensors), time.time() - start_time))
        start_time = time.time()
        outputs = self.sess.run(tensors, feed_dict=feed_dict)
        ot = [np.reshape(outputs[0], [64, 224, 224])]
        # activation_viewer.update_filter_data(ot)

        # filter_grid, filter_img = activation_viewer.draw(ot)
        print("Again predict done, %d opts. Time %.3f:" % (len(tensors), time.time() - start_time))

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
        self._get_tensor_name(tensor_type='Conv2D')
        self._get_tensor_name(tensor_type='Relu')
        if newtype != '':
            self.update_tensors(tensor_type=newtype)
        return self.tensors

    def _get_tensor_name(self, tensor_type='Conv2D'):
        '''
        Add tensor to self.tensors
        '''
        graph = self.graph
        layers = [op.name for op in graph.get_operations() if op.type == tensor_type and 'import/' in op.name]
        tensors = [graph.get_tensor_by_name(name + ':0') for name in layers]
        self.tensors.update({tensor_type: tensors})

    def update_tensors(self, loss, opts=[], gradients=None, optimizer=None):
        all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # todo, type as list
        for _var in all_vars:
            if _var not in opts:
                '''
                Some comments to understand TF
                When building ops to compute gradients,
                this op prevents the contribution of its inputs to be taken into account. '''
                tf.stop_gradient(_var)  # then the gradient only works on the input opts
        if optimizer is None:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        gradients = optimizer.compute_gradients(loss)
        optimizer.apply_gradients(gradients)

    def get_convolution_activation(self, tensor_name):
        '''
        have to know the name of the tensor
        '''

    def helper_list_tensor(self):
        pass

if __name__ == '__main__':
    '''
    Test this code with VGG16
    vgg pre-trained model can be found here:
    https://github.com/ry/tensorflow-vgg16
    '''
    # activation_viewer = SpatialActivationViewer()
    fname = "/home-local/jizha16.extra.nobkp/data/deeplearning/vgg16.tfmodel"
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        graphTF = GraphViewerTF(sess)
        graphTF.read_graph(fname)
        # feed an image to the Graph, get the responses for some ops
        img = imresize(imread('../images/lena.png'), [224, 224]) / 255.0
        img = img.reshape((1, 224, 224, 3))
        start_time = time.time()
        graphTF.forward_pass(img)
        '''
        notice the fist sess.run() will be super slow, since TF will build the graph and some sub-graphs.
        Then the graph will be cached, the subsequent computations will be fast.
        '''
        # TODO cut the a sub-graph out from the whole.
        print("Time : {}".format(time.time() - start_time))
        print 'Close'
        # with graphTF.sess:
