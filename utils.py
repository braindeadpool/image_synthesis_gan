import os
import numpy as np
from scipy.misc import imread, imresize
from scipy.io import loadmat


# Configurations
# - path settings
verbose = True  # print all debug messages
base_directory = os.path.curdir
data_directory = os.path.join(base_directory, 'data', 'sun_db')

# - training parameters
learning_rate = 0.001  # ADAM optimizer learning rate
beta1 = 0.1  # ADAM optimizer beta1

# - dataset parameters
training_data_size = 10000
batch_size = 100
image_size = [128, 128, 3]

# - variables for later usage
attribute_size = 102   # default attribute vector length

# Globally accessible dataset location specifiers
images = None
attributes = None
image_attributes = None


def is_power_of_2(num):
    """
    check's if the number is a power of 2
    :param num:
    :return: True/False
    """
    return num is not 0 and not (num & (num - 1))


def cprint(*args):
    """
    custom print wrapper
    :param args: text to print
    :return:
    """
    if verbose:
        print(args)


def get_images(image_locations, size=[256, 256, 3], base_dir=data_directory):
    """
    takes multiple image files, resizes them and returns them as a 4-D batch tensor
    :param image_locations: image locations as list of strings
    :param size: final image size
    :param base_dir: the directory relative to which the image locations are specified
    :return:
    """
    return np.array([imresize(imread(os.path.join(base_dir, image_location[0])), size)
                     for image_location in image_locations])


def load_sun_db(dir_location="./data/sun_db"):
    """
    loads the sun database mat files
    :param dir_location:
    :return: ndarray of image locations, ndarray of image attribtues, ndarray of image and corresponding attribute
    encoded vectors
    """
    global attribute_size
    global images, attributes, image_attributes
    if not os.path.isdir(dir_location):
        cprint("{} is not a valid directory! Exiting program".format(dir_location))
        exit(0)
    attributes_file = "attributes.mat"
    images_file = "images.mat"
    image_attributes_labeled_file = "attributeLabels_continuous.mat"

    # load the dataset
    images = loadmat(os.path.join(dir_location, images_file))['images'][:, 0]
    image_attributes = loadmat(os.path.join(dir_location, image_attributes_labeled_file))['labels_cv']
    attributes = loadmat(os.path.join(dir_location, attributes_file))['attributes'][:, 0]
    attribute_size = attributes.size

    # split the dataset
    images = images[:training_data_size]
    attributes = attributes[:training_data_size]
    image_attributes = image_attributes[:training_data_size]

    return images, attributes, image_attributes


if __name__ == '__main__':
    a, b, c = load_sun_db(data_directory)
    print(a)
    print(b)
    print(c)