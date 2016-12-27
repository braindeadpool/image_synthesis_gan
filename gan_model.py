#!/usr/bin/evn python
from generator import *
from discriminator import *


class GANModel(TFModel):
    def __init__(self, session=None, nid="gan", image_size=[128, 128, 3], verbose=True):
        super().__init__(session, nid, verbose)
        self._generator = Generator(self._session, output_size=image_size, verbose=verbose)
        self._discriminator = Discriminator(self._session, input_size=image_size, verbose=verbose)
        self._input_data_s_w = None  # real image data but wth incorrect attributes
        self._input_data_s_w_attributes = None  # corresponding incorrect attributes
        self._input_data_s_r = None  # real image with correct attributes
        self._input_data_s_r_attributes = None  # corresponding correct attributes
        self._d_optimizer = None  # discriminator optimizer
        self._g_optimizer = None  # generator optimizer

    def generate_samples(self, num_samples=100):
        return self._generator.generate_samples(num_samples)

    def _build_model(self):
        # add the loss function layer
        self._input_data_s_w = tf.placeholder(tf.float32,
                                              [None]+self._discriminator.input_size, name=self.nid + "_input_s_w")
        self._input_data_s_w_attributes = tf.placeholder(tf.float32,
                                              [None] + utils.attribute_size, name=self.nid + "_input_s_w_a")
        self._input_data_s_r = tf.placeholder(tf.float32,
                                              [None] + self._discriminator.input_size, name=self.nid + "_input_s_r")
        self._input_data_s_r_attributes = tf.placeholder(tf.float32,
                                                         [None] + utils.attribute_size, name=self.nid + "_input_s_r_a")

        # score is measured as the similarity between discriminator predicted attribute vector
        # and the input attribute vector
        s_w = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self._discriminator.eval(self._input_data_s_w),
                                                                     self._input_data_s_w_attributes))
        s_r = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self._discriminator.eval(self._input_data_s_r),
                                                                     self._input_data_s_r_attributes))
        generator_input = tf.placeholder(tf.float32,
                                              [None] + self._generator.input_size, name=self.nid + "_input_g")
        generator_output = self._generator.eval(generator_input)
        s_f = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self._discriminator.eval(generator_output),
                                                                     generator_input))

        # now define the discriminator loss
        d_loss = tf.log(s_r) + (tf.log(1-s_w) + tf.log(1-s_f))/2

        # now define the generator loss
        g_loss = tf.log(s_f)

        # get all trainable variables and separate them into discriminator and generator variables
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        # add the optimizers
        self._d_optimizer = tf.train.AdamOptimizer(utils.learning_rate, beta1=utils.beta1) \
            .minimize(d_loss, var_list=d_vars)
        self._g_optimizer = tf.train.AdamOptimizer(utils.learning_rate, beta1=utils.beta1) \
            .minimize(g_loss, var_list=g_vars)
        return self._g_optimizer, self._d_optimizer

    def batch_train(self, batch_input, num_epochs=1):
        self._initialize_variables()

        # now perform batch training
        with self._session as session:
            for epoch in range(0, num_epochs):
                session.run(feed_dict={self._input_data: batch_input})


if __name__ == '__main__':
    gan = GANModel()
    gan.batch_train(gan.generate_samples(10))
