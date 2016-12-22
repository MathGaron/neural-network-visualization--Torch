import tensorflow as tf
from DeepLearningBackend.BackendBase import BackendBase
import os

# test function
from scipy.misc import imread, imresize
import numpy as np


class TensorflowBackend(BackendBase):

    def backward_layer(self, grad, index):
        raise Exception("backward_layer not implemented!")

    def __init__(self, sess=None):
        tf.reset_default_graph()
        if sess is None:
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.sess = sess
        self.model = None

    def load(self, tfmodel):
        # tfmodel: any model saved from TF. /home-local/jizha16.extra.nobkp/data/ml/vgg16-tfmodel.meta
        self.model = tf.train.import_meta_graph(tfmodel)
        self.model.restore(self.sess, tf.train.latest_checkpoint(os.path.split(tfmodel)[0]))
        self.graph = tf.get_default_graph()
        # assume the first tensor is input, last tensor is the output, works for VGG
        self.output = self.get_ops()[-1]
        self.input = self.get_ops()[0]
        # set an optimizer if use backward()
        self.optimizer = None

    def forward(self, input_ims):
        # need to find out the output tensors and input tensors
        '''
        notice the fist sess.run() will be super slow, since TF will build the graph and some sub-graphs.
        Then the graph will be cached, the subsequent computations will be fast.
        '''
        self.input_ims = input_ims
        pred = self.sess.run(self.output, {self.input: input_ims})
        return pred

    def backward(self, loss, variables=[]):
        '''
        the gradient will only work on the input opts,
        if variables=[] update all the weights
        '''
        if loss is None:
            loss = self.output  # could other ops

        if self.optimizer is None:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.1)

        all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if variables == []:
            variables = all_vars

        for var in all_vars:
            if var not in variables:
                '''
                Some comments to understand TF
                When building ops to compute gradients,
                this op prevents the contribution of its inputs to be taken into account. '''
                tf.stop_gradient(var)  # then the gradient only works on the input opts
        gradients = self.optimizer.compute_gradients(loss)
        self.optimizer.apply_gradients(gradients)
        return self

    def get_activation(self):
        # run a forward pass to init the inputs
        print('get_activation')
        conv_tensors, conv_names = self.get_ops_by_type('Conv2D')
        relu_tensors, relu_names = self.get_ops_by_type('Relu')

        convos = self.sess.run(conv_tensors, {self.input: self.input_ims})
        linear = self.sess.run(relu_tensors, {self.input: self.input_ims})
        # return list of activations in shape [N, H, W, C]
        convos = tf.transpose(convos, perm=[0, 3, 2, 1])
        linear = tf.transpose(linear, perm=[0, 3, 2, 1])
        # return list as in the BackendBase,  (N, nfilters, outputsize_w, outputsize_h)
        return convos, linear

    def get_convolution_filters(self):
        all_vars = tf.trainable_variables()
        weights = {}
        for var in all_vars:
            weights.update({var.name: var.eval(session=self.sess)})
            print(var.name)
        return weights

    def get_ops(self, name=''):
        '''
        return a tensor by given name, return all if name==''
        '''
        graph = self.graph
        op_names = []
        for op in graph.get_operations():
            if (name in op.name and
                    'save' not in op.name and
                    'init' not in op.name and
                    'gradients' not in op.name):
                op_names.append(op.name)
        ops = [graph.get_tensor_by_name(op_name + ':0') for op_name in op_names]
        return ops

    def get_ops_by_type(self, op_type='Conv2D'):
        print('Get Operations: {}'.format(op_type))
        graph = self.graph
        op_names = [op.name for op in graph.get_operations() if op.type == op_type]
        ops = [graph.get_tensor_by_name(op_name + ':0') for op_name in op_names]
        return ops, op_names


if __name__ == '__main__':
    '''
    Call this function to test
    '''
    tfmodel = '/home-local/jizha16.extra.nobkp/data/ml/vgg16-tfmodel.meta'
    img = np.reshape(imresize(imread('../images/lena.png'), [224, 224]) / 255.0, [1, 224, 224, 3])
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        modelTF = TensorflowBackend(sess)
        modelTF.load_tf_model(tfmodel)
        all_vars = tf.trainable_variables()
        for v in all_vars:
            print(v.name)
        pred = modelTF.forward(img)
        print(pred)
        loss = modelTF.get_ops('pool4')  # try to compute the gradient from middle
        modelTF.backward(None, variables=[])
        print('done')
