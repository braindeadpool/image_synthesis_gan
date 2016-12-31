#!/usr/bin/env python
import tensorflow as tf


def leaky_relu(x, alpha=0.1, name="LeakyReLU"):
    """
    LeakyReLU implementation from tflearn.activations
    :param x:
    :param alpha:
    :param name:
    :return:
    """
    with tf.name_scope(name):
        x = tf.nn.relu(x)
        m_x = tf.nn.relu(-x)
        x -= alpha * m_x
    return x
