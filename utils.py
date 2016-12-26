import os
from scipy.io import loadmat


# Configurations
verbose = True  # print all debug messages
base_directory = os.path.curdir
data_directory = os.path.join(base_directory, 'data', 'sun_db')

# Variables for later usage
attribute_size = 102   # default attribute vector length


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


def load_sun_db(dir_location="./data/sun_db"):
    """
    loads the sun database mat files
    :param dir_location:
    :return: ndarray of image locations, ndarray of image attribtues, ndarray of image and corresponding attribute
    encoded vectors
    """
    global attribute_size
    if not os.path.isdir(dir_location):
        cprint("{} is not a valid directory! Exiting program".format(dir_location))
        exit(0)
    attributes_file = "attributes.mat"
    images_file = "images.mat"
    image_attributes_labeled_file = "attributeLabels_continuous.mat"
    images = loadmat(os.path.join(dir_location, images_file))
    image_with_attributes = loadmat(os.path.join(dir_location, image_attributes_labeled_file))
    attributes = loadmat(os.path.join(dir_location, attributes_file))
    attribute_size = attributes['attributes'].size
    return images['images'], attributes['attributes'], image_with_attributes['labels_cv']


if __name__ == '__main__':
    load_sun_db(data_directory)
