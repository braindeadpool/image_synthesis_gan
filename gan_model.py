#!/usr/bin/env python
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
        self._s_w = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self._discriminator_w.model,
                                                                           tf.zeros_like(self._discriminator_w.model)))
        self._s_r = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self._discriminator_r.model,
                                                                           tf.ones_like(self._discriminator_w.model)))
        self._s_f = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self._discriminator_f.model,
                                                                           tf.ones_like(self._discriminator_f.model)))

        # now define the discriminator and generator losses
        # self._d_loss = tf.log(self._s_r) - (tf.log(self._s_w) + tf.log(self._s_f))/2
        # self._g_loss = tf.log(self._s_f)

        # alternate loss functions
        self._d_loss = self._s_r - self._s_w - self._s_f
        self._g_loss = self._s_f

        # get all trainable variables and separate them into discriminator and generator variables
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        # add the optimizers
        self._d_optimizer = tf.train.AdamOptimizer(utils.learning_rate,
                                                   beta1=utils.beta1).minimize(self._d_loss, var_list=d_vars)
        self._g_optimizer = tf.train.AdamOptimizer(utils.learning_rate,
                                                   beta1=utils.beta1).minimize(self._g_loss, var_list=g_vars)

        # create the summary variables for logging and visualization
        s_w_summary = tf.scalar_summary("s_w", self._s_w)
        s_r_summary = tf.scalar_summary("s_r", self._s_r)
        s_f_summary = tf.scalar_summary("s_f", self._s_f)
        d_loss_summary = tf.scalar_summary("discriminator_loss",
                                           self._d_loss)
        g_loss_summary = tf.scalar_summary("generator_loss",
                                           self._g_loss)
        g_output = tf.image_summary("generator_output", self._generator.model, max_images=10)

        # uncomment if you have upgraded tensorflow to 0.12 and above
        # self._d_summary = tf.summary.merge([d_loss_summary, s_w_summary, s_f_summary, s_r_summary])
        # self._g_summary = tf.summary.merge([g_loss_summary, s_f_summary])
        # self._summary_writer = tf.summary.FileWriter(utils.summary_directory)

        self._d_summary = tf.merge_summary([d_loss_summary, s_w_summary, s_f_summary, s_r_summary])
        self._g_summary = tf.merge_summary([g_loss_summary, s_f_summary, g_output])

        # initialize the summary writer
        self._summary_writer = tf.train.SummaryWriter(utils.summary_directory, graph=self._session.graph)

        # initialize the saver
        self._saver = tf.train.Saver(t_vars, max_to_keep=utils.max_no_of_saves)

        return self._g_optimizer, self._d_optimizer

    def eval(self, input_data):
        return self._d_loss.eval(feed_dict=input_data), self._g_loss.eval(feed_dict=input_data)

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
                    print("Training epoch # {} / {}".format(epoch, num_epochs-1))
                for batch_num in range(0, num_batches):

                    batch_id = epoch*num_batches + batch_num
                    batch_input = sequential_batch_generator(batch_id)

                    if self._verbose:
                        print("Training batch number {}/{}, global batch number {}".format(
                            batch_num, num_batches-1, batch_id))

                    # generator input data
                    g_input = self.generate_samples(utils.batch_size)
                    feed_dict = {self._input_data_s_w: batch_input['images_s_w'],
                                 self._input_data_s_w_attributes:
                                     batch_input['images_s_w_attributes'],
                                 self._input_data_s_r: batch_input['images_s_r'],
                                 self._input_data_s_r_attributes:
                                     batch_input['images_s_r_attributes'],
                                 self._generator.input_data: g_input
                                 }
                    summary, d_opt, g_loss, d_loss = session.run([self._d_summary,
                                                                  self._d_optimizer, self._g_loss, self._d_loss],
                                                                 feed_dict=feed_dict)
                    self._summary_writer.add_summary(summary, batch_id)
                    if self._verbose:
                        print("Discriminator trained")

                    summary, g_opt, g_loss, d_loss = session.run([self._g_summary,
                                                                  self._g_optimizer, self._g_loss, self._d_loss],
                                                                 feed_dict=feed_dict)
                    self._summary_writer.add_summary(summary, batch_id)
                    if self._verbose:
                        print("Generator trained")

                    g_output = self._generator.eval(g_input)
                    utils.save_output(g_output, g_input, 'g_output_{}_{}'.format(epoch, batch_num))

                    if self._verbose:
                        print("Discriminator loss = {}".format(d_loss))
                        print("Generator loss = {}".format(g_loss))

                self._saver.save(session, 'model', global_step=epoch)
            self._summary_writer.close()


def sequential_batch_generator(batch_id=0):
    """
    generates input batches in sequence from the training data set
    :param batch_id: passing the same batch_id results in same dataset every time
    :return: batch dataset to be fed to the computational graph
    """
    batch_input = {'images_s_w': None,
                   'images_s_w_attributes': None,
                   'images_s_r': None,
                   'images_s_r_attributes': None,
                   }
    start = batch_id * utils.batch_size
    start %= utils.training_data_size
    end = min(start+utils.batch_size, utils.training_data_size)
    batch_input['images_s_w'] = utils.get_images(utils.images[start:end], utils.image_size)
    batch_input['images_s_w_attributes'] = utils.image_attributes[
                                           np.random.randint(utils.training_data_size, size=utils.batch_size), :]
    batch_input['images_s_r'] = utils.get_images(utils.images[start:end], utils.image_size)
    batch_input['images_s_r_attributes'] = utils.image_attributes[start: end, :]
    return batch_input


def random_batch_generator(batch_id=0):
    """
    generates input batches randomly from the training data set
    :param batch_id: ignored, passing the same batch_id may not result in same dataset again
    :return: batch dataset to be fed to the computational graph
    """
    batch_input = {'images_s_w': None,
                   'images_s_w_attributes': None,
                   'images_s_r': None,
                   'images_s_r_attributes': None,
                   }
    indices = np.random.randint(low=utils.training_data_size, size=utils.batch_size)
    batch_input['images_s_w'] = utils.get_images(utils.images[indices], utils.image_size)
    batch_input['images_s_w_attributes'] = utils.image_attributes[
                                           np.random.randint(utils.training_data_size, size=utils.batch_size), :]
    batch_input['images_s_r'] = utils.get_images(utils.images[indices], utils.image_size)
    batch_input['images_s_r_attributes'] = utils.image_attributes[indices, :]
    return batch_input


if __name__ == '__main__':
    # load the dataset into global variables
    utils.load_sun_db(dir_location="./data/sun_db")

    # test the random batch generator
    # batch_input = random_batch_generator(0)

    # print the parameters
    utils.print_parameters()

    gan = GANModel(image_size=utils.image_size)
    gan.train(random_batch_generator, utils.num_epochs, utils.num_batches)
