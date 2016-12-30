import os
import numpy as np
from scipy.misc import imread, imresize, imsave
from scipy.io import loadmat


# Configurations
# - path settings
verbose = True  # print all debug messages
base_directory = os.path.curdir
data_directory = os.path.join(base_directory, 'data', 'sun_db')
summary_directory = os.path.join(base_directory, 'summary')
output_directory = os.path.join(base_directory, 'output')

# - training parameters
learning_rate = 0.0005  # ADAM optimizer learning rate
beta1 = 0.1  # ADAM optimizer beta1
num_epochs = 50
num_batches = 10
max_no_of_saves = 1

# - dataset parameters
training_data_size = 10000
batch_size = 20
image_size = [128, 128, 3]

# - variables for later usage
attribute_size = 102   # default attribute vector length
noise_size = 10

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


def print_parameters():
    if verbose:
        print("Parameters\n"
              "---------------------\n"
              "learning_rate = {}\n"
              "beta1 = {}\n"
              "---------------------\n"
              "training_data_size = {}\n"
              "batch_size = {}\n"
              "num_epochs = {}\n"
              "num_batches = {}\n"
              "max_no_of_saves = {}\n"
              "---------------------\n"
              "image_size = {}\n"
              "attribute_size = {}\n"
              "noise_size = {}".format(learning_rate, beta1, training_data_size, batch_size, num_epochs,
                                       num_batches, max_no_of_saves, image_size, attribute_size, noise_size))
        

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


def save_output(image_data, input_vector, filename):
    """
    saves the generator output image and the input vector used to generate it
    :param image_data: 4-D or 3-D tensor representing batch of RGB images or RGB images
    :param input_vector: 2-D or 1-D tensor representing batch of attribute vectors or attribute vector
    :param filename: filename without extensions
    :return: filename_{batch_num}.png and filename_{batch_num}.npy will be saved
    """
    basename = os.path.join(output_directory, filename)

    if len(image_data.shape) == 3:
        imsave(basename+'.png', image_data)
        np.save(basename+'.npy', input_vector)
        attribute_vector = input_vector[:attribute_size]
        np.savetxt(basename+'.txt', attributes[np.nonzero(attribute_vector)])

        cprint("Saved output to {0}.png , {0}.npy , {0}.txt".format(basename))

    elif len(image_data.shape) == 4:
        for i in range(image_data.shape[0]):
            imsave(basename + '_{}.png'.format(i), image_data[i, :, :, :])
            np.save(basename + '_{}.npy'.format(i), input_vector[i, :])
            attribute_vector = input_vector[i, :][:attribute_size]
            with open(basename + '_{}.txt'.format(i), 'w') as f:
                for j in attributes[np.nonzero(attribute_vector)]:
                    f.write(j[0]+'\n')
            cprint("Saved output to {0}_{1}.png , {0}_{1}.npy , {0}_{1}.txt".format(basename, i))


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
    image_attributes = np.rint(image_attributes)  # convert to 0/1 representation
    attributes = loadmat(os.path.join(dir_location, attributes_file))['attributes'][:, 0]
    attribute_size = attributes.size

    # split the dataset
    images = images[:training_data_size]
    image_attributes = image_attributes[:training_data_size, :]

    return images, attributes, image_attributes


if __name__ == '__main__':
    a, b, c = load_sun_db(data_directory)
    print(a)
    print(b)
    print(c)
