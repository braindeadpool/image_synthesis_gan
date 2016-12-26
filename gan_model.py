#!/usr/bin/evn python
from generator import *
from discriminator import *


class GANModel(TFModel):
    def __init__(self, session=None, nid="gan", verbose=True):
        super().__init__(session, nid, verbose)
        self._generator = Generator(self._session, verbose=verbose)
        self._discriminator = Discriminator(self._session, verbose=verbose)

    def generate_samples(self, num_samples=100):
        return self._generator.generate_samples(num_samples)

    def batch_train(self, batch_input):
        self._initialize_variables()
        # now perform batch training
        with self._session as session:
            session.run(feed_dict={self._input_data: batch_input})


if __name__ == '__main__':
    gan = GANModel()
    gan.batch_train(gan.generate_samples(10))
