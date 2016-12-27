#!/usr/bin/evn python
import utils
import numpy as np
from tfmodel import *


class Generator(TFModel):
    """
    Generator network
    """
    def __init__(self, session=None, nid="g", noise_size=10, output_size=[128, 128, 3], verbose=True, reuse=False):
        super().__init__(session, nid, verbose, reuse)
        self.noise_size = noise_size   # noise vector size
        self.attribute_size = utils.attribute_size
        self.input_size = self.attribute_size + self.noise_size  # final input vector size
        self.output_size = output_size  # output image size
        self.mu = 0    # normal distribution mean
        self.sigma = 1    # normal distribution std
        self._build_model()

    def generate_samples(self, num_samples=100):
        """
        samples from a random distribution to generate a attribute+noise vector encoding
        :num_samples: number of vector samples to generate
        :return:
        """
        samples = self.mu + np.random.randn(num_samples, self.input_size) * self.sigma
        # convert the attribute vector to binary representation
        samples[:, self.attribute_size] /= self.sigma
        samples[:, self.attribute_size] = np.rint(samples[:, self.attribute_size])
        return samples

    def _build_model(self):
        """
        tensorflow graph model of the generator network
        :return:
        """
        fc0_std = 0.2  # std of random initialization for fc0 matrix
        fc0_shape = [4, 4, 1024]  # shape of the fc0 reshaped output (for batch size = 1)
        fc0_size = fc0_shape[0] * fc0_shape[1] * fc0_shape[2]  # size of the fc0 output
        self._input_data = tf.placeholder(tf.float32, [None, self.input_size], name=self.nid + "_input")
        # project and reshape the input array - basically an fc layer doing matrix multiplication and bias addition
        with tf.variable_scope(self.nid+"fc0"):
            W = tf.get_variable("W", shape=[self.input_size, fc0_size], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=fc0_std))
            b = tf.get_variable("b", shape=[fc0_size], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=fc0_std))
            fc0_output = tf.reshape(tf.matmul(self._input_data, W) + b, [-1] + fc0_shape)

        fsconv_input = fc0_output  # initial fsconv input is the output of the fc layer

        # filter's first 2 dimensions, the rest two are auto computed
        filter_shape = [5, 5]
        # generate output shapes for each fsconv layer
        output_shapes = [[int(2 ** x), int(2 ** x), int(fc0_shape[2] * 4 / (2 ** x))]
                         for x in range(3, 10) if int(2**x) <= np.min(self.output_size[:2])]
        # set the last output shape to be 3-channeled (or as required by the model)
        output_shapes[-1][2] = self.output_size[2]
        if self._verbose:
            print("FSConv layer output shapes - {}".format(output_shapes))

        # create the intermediate fsconv layers
        for output_shape in output_shapes:
            with tf.variable_scope(self.nid+"_fsconv-{}x{}".format(output_shape[0], output_shape[1])):
                W_shape = filter_shape+[output_shape[2]]+[fsconv_input.get_shape().as_list()[-1]]
                W = tf.get_variable("W", initializer=tf.truncated_normal(W_shape, stddev=0.1))
                # fractionally-strided convolution network
                fsconv = tf.nn.conv2d_transpose(fsconv_input, W, output_shape=[-1]+output_shape, strides=[1, 2, 2, 1])
                fsconv = tf.nn.relu(fsconv)     # apply relu layer
            fsconv_input = fsconv

        self._model = fsconv

    def eval(self, input_data):
        self._initialize_variables()
        with self._session:
            return self._model.eval(feed_dict={self._input_data: input_data})


if __name__ == '__main__':
    # create generator and test some methods
    g = Generator()
    print(g.generate_samples(10))
