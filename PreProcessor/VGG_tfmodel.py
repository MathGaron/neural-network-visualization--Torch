import os
import numpy as np
import tensorflow as tf
import time

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg16:

    def __init__(self, vgg16_npy_path=None, trainable=True):
        if vgg16_npy_path is None:
            self.data_dict = {}
        else:
            assert os.path.isfile(vgg16_npy_path), vgg16_npy_path + " doesn't exist."
            self.data_dict = np.load(vgg16_npy_path).item()
            print "npy file loaded"
        self.var_dict = {}
        self.trainable = trainable

    def build(self, rgb):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """
        #  if True, dropout will be turned on
        self.training = tf.Variable(True, trainable=False, name="training")

        start_time = time.time()
        print "build model started"
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(3, 3, rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(3, [
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6, name="relu6")
        #//
        self.keep_prob = tf.cond(self.training, lambda: tf.constant(0.5), lambda: tf.constant(1.0), name="keep_prob")
        self.drop6 = tf.nn.dropout(self.relu6, self.keep_prob, name="drop6")

        self.fc7 = self.fc_layer(self.drop6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7, name="relu7")
        #//
        self.drop7 = tf.nn.dropout(self.relu7, self.keep_prob, name="drop7")

        self.fc8 = self.fc_layer(self.drop7, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None
        print "build model finished: %ds" % (time.time() - start_time)

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(bottom, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(bottom, name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(x, name)
            biases = self.get_bias(x, name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_n_out(self, name):
        if name[:4] == 'conv':
            n_out = 64 * (2 ** (min(int(name[4]), 4) - 1))
        else:
            if name[2] == '8':
                n_out = 1000
            else:
                n_out = 4096
        return n_out

    def get_conv_filter(self, bottom, name):
        if self.data_dict.get(name, None) is None:
            print 'No pretrained weight for', name, 'filter'
            n_in = bottom.get_shape()[-1].value
            n_out = self.get_n_out(name)
            print 'n_in', n_in, 'n_out', n_out
            return tf.get_variable("filter",
                                   shape=[3, 3, n_in, n_out],
                                   dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer_conv2d())
        return tf.Variable(self.data_dict[name][0], name="filter")

    def get_bias(self, bottom, name):
        if self.data_dict.get(name, None) is None:
            print 'No pretrained weight for', name, 'biases'
            n_out = self.get_n_out(name)
            print 'n_out', n_out
            return tf.Variable(tf.constant(0.0, shape=[n_out], dtype=tf.float32), trainable=True, name='biases')
        return tf.Variable(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, bottom, name):
        if self.data_dict.get(name, None) is None:
            print 'No pretrained weight for', name, 'weights'
            n_in = bottom.get_shape()[-1].value
            n_out = self.get_n_out(name)
            print 'n_in', n_in, 'n_out', n_out
            return tf.get_variable("weights",
                                   shape=[n_in, n_out],
                                   dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer())
        return tf.Variable(self.data_dict[name][0], name="weights")


if __name__ == '__main__':
    '''
    Call this function to save the pre-trained VGG16 as tensorflow model
    Download pretrained model here:
    https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM
    '''
    fname = '/home-local/jizha16.extra.nobkp/data/ml/vgg16.npy'
    images = tf.placeholder(tf.float32, [None, 224, 224, 3])

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        modelVGG = Vgg16(fname)
        modelVGG.build(images)
        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())
        saver.save(sess, '/home-local/jizha16.extra.nobkp/data/ml/vgg16-tfmodel')
        # `save` method will call `export_meta_graph` implicitly.
        # you will get saved graph files:my-model.meta
        # we could save/resotre any tf model

    print 'save model to vgg16-tfmodel.meta'

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        loader = tf.train.import_meta_graph('/home-local/jizha16.extra.nobkp/data/ml/vgg16-tfmodel.meta')
        loader.restore(sess, tf.train.latest_checkpoint('/home-local/jizha16.extra.nobkp/data/ml/'))
        all_vars = tf.trainable_variables()
        for v in all_vars:
            print(v.name)
