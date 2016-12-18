#!/usr/bin/evn python
import utils
import numpy as np
import tensorflow as tf


class Generator:
    """
    Generator tensorflow implementation
    """
    def __init__(self):
        self.noise_size = 10   # noise vector size
        self.attribute_size = utils.attribute_size
        self.input_size = self.attribute_size + self.noise_size  # final input vector size
        self.mu = 0    # normal distribution mean
        self.sigma = 1    # normal distribution std

    def generate_samples(self, num_samples=100):
        """
        samples from a random distribution to generate a attribute+noise vector encoding
        :num_samples: number of vector samples to generate
        :return:
        """
        return self.mu + np.rand(1, self.input_size) * self.sigma
