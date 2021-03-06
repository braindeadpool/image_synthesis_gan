#!/usr/bin/env python
import utils
import addons
import numpy as np
from tfmodel import *


class Discriminator(TFModel):
    """
    Discriminator network
    """

    def __init__(self, session=None, nid="d", input_data=None, input_attributes = None, input_size=[64, 64, 3], verbose=True, reuse=False):
        super().__init__(session, nid, input_data, verbose, reuse)
        self.input_size = input_size
        self._input_attributes = input_attributes
        self._build_model()

    def _build_model(self):
        """
        tensorflow graph model of the discriminator network
        :return:
        """
        # if input is not a variable connecting the discriminator to another network (ex, generator output),
        # initialize a placeholder
        if self._input_data is None:
            # placeholder for input data
            self._input_data = tf.placeholder(tf.float32, [None]+self.input_size, name=self.nid + "_input")

        # filter's first 2 dimensions, the rest two are auto computed
        filter_shape = [5, 5]
        # generate output shapes for each conv layer
        output_shapes = [[int(self.input_size[0]/2**x), int(self.input_size[1]/2**x), int(32 * 2 ** x)]
                         for x in range(1, 6)]
        # size of the pre-fc layer
        fc0_size = output_shapes[-1][0] * output_shapes[-1][1] * output_shapes[-1][2]
        # std of the fc0 initializer
        fc0_std = 0.2

        conv_input = self._input_data
        conv_shapes = []

        # create the conv layers
        for output_shape in output_shapes:
            with tf.variable_scope(self.nid+"_conv-{}x{}".format(output_shape[0], output_shape[1]), reuse=self._reuse):
                W_shape = filter_shape+[conv_input.get_shape().as_list()[-1]]+[output_shape[2]]
                W = tf.get_variable("W", initializer=tf.truncated_normal(W_shape, stddev=0.1))
                b = tf.get_variable("b", shape=output_shape[-1:], initializer=tf.constant_initializer(0.0))
                # convolution network
                conv = tf.nn.conv2d(conv_input, W, strides=[1, 2, 2, 1], padding='SAME')
                conv = tf.nn.bias_add(conv, b)
                conv = addons.leaky_relu(conv, 0.2)     # apply relu layer
                # store the shape
                conv_shapes.append(conv.get_shape().as_list())
            conv_input = conv

        # From the text to speech gan paper (not used in the final version of this)
        # if input attributes is provided, spatially merge it with the reshaped conv output
        # if self._input_attributes is not None:
        #     batch_size = tf.shape(conv)[0]
        #     conv = tf.reshape(conv, [batch_size, fc0_size])
        #     conv = tf.concat(1, [conv, self._input_attributes])
        #     # now update fc0_size with based on the new conv shape so that the final fc layer is created correctly
        #     fc0_size += utils.attribute_size

        # create the final fc layer
        with tf.variable_scope(self.nid+"_fc0", reuse=self._reuse):
            W = tf.get_variable("W", shape=[fc0_size, utils.attribute_size], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=fc0_std))
            b = tf.get_variable("b", shape=[utils.attribute_size], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=fc0_std))
            conv = tf.reshape(conv, [-1, fc0_size])
            fc0_output = tf.nn.bias_add(tf.matmul(conv, W), b)
            fc0_output = tf.nn.relu(fc0_output)

        if self._verbose:
            print("Conv layer output shapes - {}".format(conv_shapes+[fc0_output.get_shape().as_list()]))

        self._model = fc0_output

    def eval(self, input_data):
        self._initialize_variables()
        return self._model.eval(feed_dict={self._input_data: input_data})


if __name__ == '__main__':
    # create generator and test some methods
    d = Discriminator()
