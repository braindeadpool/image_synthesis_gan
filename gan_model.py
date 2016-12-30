#!/usr/bin/evn python
import os
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
            self._discriminator_f.model, tf.slice(self._generator.input_data,
                                                  [0, 0], [-1, self._generator.attribute_size])))

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

        # create the summary variables for logging and visualization
        s_w_summary = tf.summary.scalar("s_w", s_w, "Discriminator score with real data but incorrect attributes")
        s_r_summary = tf.summary.scalar("s_r", s_r, "Discriminator score with real data and correct attributes")
        s_f_summary = tf.summary.scalar("s_f", s_f, "Discriminator score with fake data and attributes")
        d_loss_summary = tf.summary.scalar("discriminator_loss",
                                           d_loss,
                                           "The loss function tried to minimize by the discriminator network")
        g_loss_summary = tf.summary.scalar("generator_loss",
                                           d_loss,
                                           "The loss function tried to minimize by the generator network")

        # uncomment if you have upgraded tensorflow to 0.12 and above
        # self._d_summary = tf.summary.merge([d_loss_summary, s_w_summary, s_f_summary, s_r_summary])
        # self._g_summary = tf.summary.merge([g_loss_summary, s_f_summary])
        # self._summary_writer = tf.summary.FileWriter(utils.summary_directory)

        self._d_summary = tf.merge_summary([d_loss_summary, s_w_summary, s_f_summary, s_r_summary])
        self._g_summary = tf.merge_summary([g_loss_summary, s_f_summary])

        # initialize the summary writer
        self._summary_writer = tf.train.SummaryWriter(utils.summary_directory, graph=self._session.graph)

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
                if self._verbose:
                    print("Training epoch # {} / {}".format(epoch, num_epochs))
                for batch_num in range(0, num_batches):

                    batch_id = epoch*num_batches + batch_num % utils.training_data_size
                    batch_input = batch_generator(batch_id)

                    if self._verbose:
                        print("Training batch number {}/{}, global batch number {}".format(
                            batch_num, num_batches, batch_id))

                    # generator input data
                    g_input = self.generate_samples(utils.batch_size)
                    summary, _ = session.run([self._d_summary,
                                              self._d_optimizer],
                                             feed_dict={self._input_data_s_w: batch_input['images_s_w'],
                                                        self._input_data_s_w_attributes:
                                                        batch_input['images_s_w_attributes'],
                                                        self._input_data_s_r: batch_input['images_s_r'],
                                                        self._input_data_s_r_attributes:
                                                        batch_input['images_s_r_attributes'],
                                                        self._generator.input_data: g_input
                                                        })
                    self._summary_writer.add_summary(summary, batch_id)
                    if self._verbose:
                        print("Discriminator trained")

                    summary, _ = session.run([self._g_summary, self._g_optimizer],
                                             feed_dict={
                                                 self._generator.input_data: g_input})
                    self._summary_writer.add_summary(summary, batch_id)
                    if self._verbose:
                        print("Generator trained")

                    g_output = self._generator.eval(g_input)
                    utils.save_output(g_output, 'g_output_{}_{}'.format(epoch, batch_num))

        self._summary_writer.flush()
        self._summary_writer.close()


def batch_generator(batch_id=0):
    batch_input = {'images_s_w': None,
                   'images_s_w_attributes': None,
                   'images_s_r': None,
                   'images_s_r_attributes': None,
                   }
    start = batch_id * utils.batch_size
    batch_input['images_s_w'] = utils.get_images(utils.images[start:start + utils.batch_size], utils.image_size)
    batch_input['images_s_w_attributes'] = utils.image_attributes[
                                           np.random.randint(utils.training_data_size, size=utils.batch_size), :]
    batch_input['images_s_r'] = utils.get_images(utils.images[start:start + utils.batch_size], utils.image_size)
    batch_input['images_s_r_attributes'] = utils.image_attributes[start: start+utils.batch_size, :]
    return batch_input


if __name__ == '__main__':
    gan = GANModel(image_size=utils.image_size)

    # load the dataset into global variables
    utils.load_sun_db(dir_location="./data/sun_db")
    gan.train(batch_generator, 1, 1)
