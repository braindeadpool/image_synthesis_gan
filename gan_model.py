#!/usr/bin/evn python
from generator import *
from discriminator import *
tf.logging.set_verbosity(tf.logging.INFO)


class GANModel(TFModel):
    def __init__(self, session=None, nid="gan", image_size=[128, 128, 3], verbose=True):
        super().__init__(session, nid, verbose)
        self._image_size = image_size
        # initialize a generator
        self._generator = None
        # we need a discriminator object for each w, r, f dataset - all reuse variables
        self._discriminator_w = None
        self._discriminator_r = None
        self._discriminator_f = None
        self._input_data_s_w = None  # real image data but wth incorrect attributes
        self._input_data_s_w_attributes = None  # corresponding incorrect attributes
        self._input_data_s_r = None  # real image with correct attributes
        self._input_data_s_r_attributes = None  # corresponding correct attributes
        self._d_optimizer = None  # discriminator optimizer
        self._g_optimizer = None  # generator optimizer
        self._build_model()

    def generate_samples(self, num_samples=100):
        return self._generator.generate_samples(num_samples)

    def _build_model(self):
        # add the loss function layer
        self._input_data_s_w = tf.placeholder(tf.float32,
                                              [None]+self._image_size, name=self.nid + "_input_s_w")
        self._input_data_s_w_attributes = tf.placeholder(tf.float32,
                                              [None, utils.attribute_size], name=self.nid + "_input_s_w_a")
        self._input_data_s_r = tf.placeholder(tf.float32,
                                              [None] + self._image_size, name=self.nid + "_input_s_r")
        self._input_data_s_r_attributes = tf.placeholder(tf.float32,
                                                         [None, utils.attribute_size], name=self.nid + "_input_s_r_a")

        # initialize a generator
        self._generator = Generator(self._session, output_size=self._image_size, verbose=self._verbose)
        # we need a discriminator object for each w, r, f dataset - all reuse variables
        self._discriminator_w = Discriminator(self._session, input_data=self._input_data_s_w,
                                              input_size=self._image_size, verbose=self._verbose, reuse=False)
        self._discriminator_r = Discriminator(self._session, input_data=self._input_data_s_r,
                                              input_size=self._image_size, verbose=self._verbose, reuse=True)
        self._discriminator_f = Discriminator(self._session, input_data=self._generator.model,
                                              input_size=self._image_size, verbose=self._verbose, reuse=True)

        # score is measured as the similarity between discriminator predicted attribute vector
        # and the input attribute vector
        # generator/discriminator.model = score for the input passed
        s_w = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self._discriminator_w.model,
                                                                     self._input_data_s_w_attributes))
        s_r = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self._discriminator_r.model,
                                                                     self._input_data_s_r_attributes))
        # we slice the input to generator to separate out only the attribute vector component (ie ignore the noise part)
        s_f = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            self._discriminator_f.model, tf.slice(self._generator.input_data, [0, self._generator.noise_size], [-1, -1])))

        # now define the discriminator loss
        d_loss = tf.log(s_r) + (tf.log(1-s_w) + tf.log(1-s_f))/2

        # now define the generator loss
        g_loss = tf.log(s_f)

        # get all trainable variables and separate them into discriminator and generator variables
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        # add the optimizers
        self._d_optimizer = tf.train.AdamOptimizer(utils.learning_rate,
                                                   beta1=utils.beta1).minimize(d_loss, var_list=d_vars)
        self._g_optimizer = tf.train.AdamOptimizer(utils.learning_rate,
                                                   beta1=utils.beta1).minimize(g_loss, var_list=g_vars)
        return self._g_optimizer, self._d_optimizer

    def train(self, batch_generator, num_epochs=1, num_batches=10):
        """
        train the GAN
        :param batch_generator: a method that returns a batch of input data - optionally takes batch number (batch id)
        as a parameter
        :param num_epochs:
        :param num_batches:
        :return:
        """
        self._initialize_variables()

        # now perform batch training
        with self._session as session:
            for epoch in range(0, num_epochs):
                for batch_num in range(0, num_batches):
                    batch_id = epoch*num_batches + batch_num % utils.training_data_size
                    batch_input = batch_generator(batch_id)
                    session.run([self._d_optimizer],
                                feed_dict={self._input_data_s_w: batch_input['images_s_w'],
                                           self._input_data_s_w_attributes:
                                               batch_input['images_s_w_attributes'],
                                           self._input_data_s_r: batch_input['images_s_r'],
                                           self._input_data_s_r_attributes:
                                               batch_input['images_s_r_attributes']
                                           })
                    session.run([self._g_optimizer],
                                feed_dict={self._generator.input_data: self.generate_samples(utils.batch_size)})


def batch_generator(batch_id=0):
    batch_input = {'images_s_w': None,
                   'images_s_w_attributes': None,
                   'images_s_r': None,
                   'images_s_r_attributes': None,
                   }
    start = batch_id * utils.batch_size
    batch_input['images_s_w'] = utils.get_images(utils.images[start:start + utils.batch_size], utils.image_size)
    batch_input['images_s_w_attributes'] = utils.image_attributes[
                                           np.random.randint(0, utils.attribute_size, utils.batch_size), :]
    batch_input['images_s_r'] = utils.get_images(utils.images[start:start + utils.batch_size], utils.image_size)
    batch_input['images_s_r_attributes'] = utils.image_attributes[start: start+utils.batch_size, :]
    return batch_input


if __name__ == '__main__':
    gan = GANModel()

    # load the dataset into global variables
    utils.load_sun_db(dir_location="./data/sun_db")
    gan.train(batch_generator, 1, 1)
