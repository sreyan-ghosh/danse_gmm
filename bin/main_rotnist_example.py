##################################################
## Project: RotNIST
## Script purpose: To download MNIST dataset and append new rotated digits to it
## Date: 21st April 2018
## Author: Chaitanya Baweja, Imperial College London
##################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gzip
import os
import numpy as np
import tensorflow as tf
import time
import csv
from scipy import ndimage
from six.moves import urllib
from PIL import Image
# from scipy.misc import imsave
import imageio

#Url for downloading MNIST dataset
URL = 'http://yann.lecun.com/exdb/mnist/'
#Data Directory where all data is saved
DATA_DIRECTORY = "data"

# Params for MNIST

VALIDATION_SIZE = 5000  # Size of the validation set.

'''
Download the data from Yann's website, unless it's already here.
filename: filepath to images
Returns path to file
'''
def download(filename):
    #Check if directory exists
    if not tf.io.gfile.exists(DATA_DIRECTORY):
        tf.io.gfile.makedirs(DATA_DIRECTORY)
    filepath = os.path.join(DATA_DIRECTORY, filename)
    #Check if file exists, if not download
    if not tf.io.gfile.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(URL + filename, filepath)
        with tf.io.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath
'''
Extract images from given file path into a 4D tensor [image index, y, x, channels].
filename: filepath to images
num: number of images
60000 in case of training
10000 in case of testing
Returns numpy vector
'''
def extract_data(filename, num):
    print('Extracting', filename)
    #unzip data
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * num * 1)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num, 28, 28, 1) #reshape into tensor
        data = np.reshape(data, [num, -1])
    return data

'''
Extract the labels into a vector of int64 label IDs.
filename: filepath to labels
num: number of labels
60000 in case of training
10000 in case of testing
Returns numpy vector
'''
def extract_labels(filename, num):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        num_labels_data = len(labels)
        one_hot_encoding = np.zeros((num_labels_data,10))
        one_hot_encoding[np.arange(num_labels_data),labels] = 1
        one_hot_encoding = np.reshape(one_hot_encoding, [-1, 10])
    return one_hot_encoding

'''
Augment training data with rotated digits
images: training images
labels: training labels
'''
def expand_training_data(images, labels, T=10):

    expanded_images = []
    expanded_labels = []
    directory = os.path.dirname("data/New")
    if not tf.io.gfile.exists("data/New"):
        tf.io.gfile.makedirs("data/New")
    k = 0 # counter
    for x, y in zip(images, labels):
        k = k+1
        if k%100==0:
            print ('expanding data : %03d / %03d' % (k,np.size(images,0)))

        # register original data
        expanded_images.append(x)
        expanded_labels.append(y)

        bg_value = 0 # this is regarded as background's value black

        image = np.reshape(x, (-1, 28))
        for i in range(T):
            # Rotate angle depending on length T
            angle = int(360/T*(i+1))
            new_img = ndimage.rotate(image,angle,reshape=False, cval=bg_value)

            #code for saving some of these for visualization purpose only
            image1 = (image*255) + (255 / 2.0)
            new_img1 = (new_img*255) + (255 / 2.0)

            if k<50:
                NAME1 = DATA_DIRECTORY+"/New"+"/"+str(k)+"_0.jpeg"
                im = Image.fromarray(image1)
                im.convert('RGB').save(NAME1)
                im = Image.fromarray(new_img1)
                NAME = DATA_DIRECTORY+"/New"+"/"+str(k)+"_"+str(i+1)+".jpeg"
                im.convert('RGB').save(NAME)

            # register new training data
            expanded_images.append(np.reshape(new_img, 784))
            expanded_labels.append(y)


    # images and labels are concatenated
    # notice that pair of image and label should not be broken
    expanded_train_total_data = np.concatenate((expanded_images, expanded_labels), axis=1)

    return expanded_train_total_data

'''
Main function to prepare the entire data
'''
def prepare_MNIST_data(use_data_augmentation=True):
    # Get the data.
    train_data_filename = download('train-images-idx3-ubyte.gz')
    train_labels_filename = download('train-labels-idx1-ubyte.gz')
    test_data_filename = download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = download('t10k-labels-idx1-ubyte.gz')

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, 60000)
    train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)
    # Generate a validation set.
    print(train_data.shape)
    validation_data = train_data[:VALIDATION_SIZE, :]
    validation_labels = train_labels[:VALIDATION_SIZE,:]
    train_data = train_data[VALIDATION_SIZE:, :]
    train_labels = train_labels[VALIDATION_SIZE:,:]

    # Concatenate train_data & train_labels for random shuffle
    if use_data_augmentation:
        train_total_data = expand_training_data(train_data, train_labels)
    else:
        train_total_data = np.concatenate((train_data, train_labels), axis=1)

    train_size = train_total_data.shape[0]

    return train_total_data, train_size, validation_data, validation_labels, test_data, test_labels
train_total_data, train_size, validation_data, validation_labels, test_data, test_labels = prepare_MNIST_data(True)
