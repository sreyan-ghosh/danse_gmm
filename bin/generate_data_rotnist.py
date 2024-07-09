##################################################
## Project: RotNIST for DANSE
## Script purpose: To download MNIST dataset and append new rotated digits to it
## Stores the images as jpg files with a CSV for labels
## Date: 04 July 2024
## Author: Chaitanya Baweja, Sreyan Ghosh, Kasper Malm
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
import argparse
from parse import parse
from scipy import ndimage
from six.moves import urllib
from PIL import Image
# from scipy.misc import imsave
from imageio import imwrite


#Url for downloading MNIST dataset
URL = 'http://yann.lecun.com/exdb/mnist/'
#Data Directory where all data is saved
DATA_DIRECTORY = "data"


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
Extract images from given file path into a 3D tensor [image index, y, x].
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
        buf = bytestream.read(28 * 28 * num)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num, 28, 28, 1) #reshape into tensor
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
        buf = bytestream.read(num)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

'''
Augment training data with rotated digits
images: training images
labels: training labels
'''
def expand_training_data(images, labels, T=10):

    expanded_images = []
    expanded_labels = []
    
    k = 0 # counter
    for x, y in zip(images, labels):
        #print(x.shape)
        k = k+1
        if k%100==0:
            print ('expanding data : %03d / %03d' % (k,np.size(images,0)))

        # register original data
        expanded_images.append(x)
        expanded_labels.append(y)

        bg_value = 0 # this is regarded as background's value black
        #print(x)
        image = np.reshape(x, (-1, 28))
        
        for i in range(T-1):
            # rotate the image with random degree
            angle = int(360/T)*(i+1)
            rotated_padded_img = ndimage.rotate(image, angle, reshape=False, cval=bg_value, order=3)

            new_img = np.clip(rotated_padded_img, 0, 255)


            # register new training data
            expanded_images.append(np.reshape(new_img, (28, 28, 1)))
            expanded_labels.append(y)

    # return them as arrays

    expandedX=np.asarray(expanded_images)
    expandedY=np.asarray(expanded_labels)
    return expandedX, expandedY

# Prepare MNISt data
def prepare_MNIST_data(use_data_augmentation=True, T=None, N=None, output_path=None):
    # Get the data.
    train_data_filename = download('train-images-idx3-ubyte.gz')
    train_labels_filename = download('train-labels-idx1-ubyte.gz')
    test_data_filename = download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = download('t10k-labels-idx1-ubyte.gz')

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, N)
    train_labels = extract_labels(train_labels_filename, N)
    test_data = extract_data(test_data_filename, int(N*0.1))
    test_labels = extract_labels(test_labels_filename, int(N*0.1))

    DATADIR = output_path

    if use_data_augmentation:
        train_data, train_labels = expand_training_data(train_data, train_labels)

    if not os.path.isdir(os.path.join(DATADIR, "train-images")):
        os.makedirs(os.path.join(DATADIR, "train-images"))
    
    # process train data
    with open(os.path.join(DATADIR, "train-labels.csv"), 'w') as csvFile:
        writer = csv.writer(csvFile, delimiter=',', quotechar='"')
        imnum, j = 0, 0
        for i in range(len(train_data)):
            if i%T == 0:
                imnum += 1
                j = 0
            j += 1
            array = train_data[i][:,:,0]
            array = array.astype(np.uint8)
            cur_img = Image.fromarray(array)
            gray_img = cur_img.convert("L")
            imwrite(DATADIR + "/train-images/" + f"{imnum}_{j}" + ".jpg", gray_img)
            writer.writerow(["train-images/" + f"{imnum}_{j}" + ".jpg", train_labels[i]])

    # repeat for test data
    # with open(f"{output_path}/test-labels.csv", 'w') as csvFile:
    #     writer = csv.writer(csvFile, delimiter=',', quotechar='"')
    #     imnum, j = 0, 0
    #     for i in range(len(test_data)):
    #         if i%T == 0:
    #             imnum += 1
    #             j = 0
    #         j += 1
    #         array = test_data[i][:,:,0]
    #         array = array.astype(np.uint8)
    #         cur_img = Image.fromarray(array)
    #         gray_img = cur_img.convert("L")
    #         imwrite(DATADIR + "/test-images/" + f"{imnum}_{j}" + ".jpg", gray_img)
    #         writer.writerow(["test-images/" + f"{imnum}_{j}" + ".jpg", test_labels[i]])
    #return train_total_data, train_size, validation_data, validation_labels, test_data, test_labels
    
if __name__ == "__main__":
    
    # Parse shell script
    #---------------------------------------------------
    parser = argparse.ArgumentParser(description="Input arguments related to creating a dataset for ROTNIST")
    
    # N
    parser.add_argument("--num_samples", help="denotes the number of trajectories to be simulated for each realization", type=int, default=1000)
    # T
    parser.add_argument("--sequence_length", help="denotes the length of each trajectory", type=int, default=10)
    parser.add_argument("--output_path", help="Enter full path to store the data file", type=str, default="data/rotnist")

    args = parser.parse_args() 


    T = args.sequence_length
    N_samples = args.num_samples
    output_path = args.output_path

    """
    # Create the full path for the datafile
    datafilename = create_filename(T=T, N_samples=N_samples, dataset_basepath=output_path)


    # If the dataset hasn't been already created, create the dataset
    if not os.path.isfile(datafilename):
        print("Creating the data file: {}".format(datafilename))
        create_and_save_dataset(T=T, N_samples=N_samples, filename=datafilename)

    else:
        print("Dataset {} is already present!".format(datafilename))

    print("Done...")
    """
    #---------------------------------------------------
        
    prepare_MNIST_data(True, T, N_samples, output_path)
