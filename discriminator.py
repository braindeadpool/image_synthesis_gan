#!/usr/bin/evn python
import utils
import numpy as np
from tfmodel import *


class Discriminator(TFModel):
    """
    Discriminator network
    """

    def __init__(self, session=None, nid="d", input_size=[64, 64, 3], verbose=True):
        super().__init__(session, nid, verbose)
        self.input_size = input_size
        self._build_model()

    def _build_model(self):
        """
        tensorflow graph model of the discriminator network
        :return:
        """
        self._input_data = tf.placeholder(tf.float32, [None]+self.input_size, name=self.nid + "_input")

        # filter's first 2 dimensions, the rest two are auto computed
        filter_shape = [5, 5]

        # generate output shapes for each conv layer
        output_shapes = [[int(self.input_size[0]/2**x), int(self.input_size[1]/2**x), int(32 * 2 ** x)]
                         for x in range(1, 5)]

        # add final fc layer output shape
        output_shapes.append([1, 1, 1])
        if self.verbose:
            print("Conv layer output shapes - {}".format(output_shapes))

        conv_input = self._input_data

        # create the conv layers
        for output_shape in output_shapes:
            with tf.variable_scope(self.nid+"_conv-{}x{}".format(output_shape[0], output_shape[1])):
                W_shape = filter_shape+[conv_input.get_shape().as_list()[-1]]+[output_shape[2]]
                W = tf.get_variable("W", initializer=tf.truncated_normal(W_shape, stddev=0.1))
                # convolution network
                conv = tf.nn.conv2d(conv_input, W, strides=[1, 2, 2, 1], padding='SAME')
                conv = tf.nn.relu(conv)     # apply relu layer
            conv_input = conv

        self._model = conv

    def eval(self, input_data):
        self._initialize_variables()
        with self._session:
            return self._model.eval(feed_dict={self._input_data: input_data})


if __name__ == '__main__':
    # create generator and test some methods
    d = Discriminator()
