import fnmatch
import os
import tensorflow as tf
import numpy as np
import PIL
import matplotlib.pyplot as plt

from PIL import Image

TRAIN_PATH = './Data/Training/'
TRAIN_DATA_NAME = 'trainData.tfrecords'

TEST_PATH = './Data/Test/'
TEST_DATA_NAME = 'testData.tfrecord'

DATA_LABELS = ['lever', 'push', 'clothes', 'dial', 'sliding','onOff']
NUM_OF_LABELS = 6

#harcoding image sizels
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 120
COLOR_NUM_OF_CHANNELS = 3
DEPTH_NUM_OF_CHANNELS = 1
COL_IMAGE_SIZE = [IMAGE_WIDTH, IMAGE_HEIGHT, COLOR_NUM_OF_CHANNELS]
COL_IMAGE_SIZE_FLAT = IMAGE_WIDTH *  IMAGE_HEIGHT * COLOR_NUM_OF_CHANNELS

DEP_IMAGE_SIZE = [IMAGE_WIDTH, IMAGE_HEIGHT, DEPTH_NUM_OF_CHANNELS]
DEP_IMAGE_SIZE_FLAT = IMAGE_WIDTH *  IMAGE_HEIGHT * DEPTH_NUM_OF_CHANNELS

#some prodecures
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
##########################################################################
def loadLabelsAndFilenames(directoryAddress):
    
    colorAddresses = []
    depthAddresses = []
    for root, dirnames, filenames in os.walk(directoryAddress):     
        print(dirnames)   
        jpgMatch = fnmatch.filter(filenames,'*.jpg')
        
        for filename in jpgMatch:
            if "depth" in filename:
                depthAddresses.append(os.path.join(root,filename))
            else:
                colorAddresses.append(os.path.join(root,filename))

    colorAddresses.sort()
    depthAddresses.sort()



    colorLabels = []
    for address in colorAddresses:
        for i in range(0,NUM_OF_LABELS):
            if DATA_LABELS[i] in address:
                colorLabels.append(i)

    depthLabels = []
    for address in depthAddresses:
         for i in range(0,NUM_OF_LABELS):
            if DATA_LABELS[i] in address:
                depthLabels.append(i)


    numOfData = len(depthLabels) if len(depthLabels)==len(colorLabels) else 0 
    print("num of data : {}".format(numOfData))
    return colorAddresses, colorLabels, depthAddresses, depthLabels, numOfData

def writeTFrecord(dirAddress, tfFilename):
    
    colorAddress, colorLabels, depthAddress, depthLabels, numOfData = loadLabelsAndFilenames(dirAddress)
    
    numOfColorImages = len(colorAddress) if len(colorAddress) == len(colorLabels) else 0
    numOfDepthImages = len(depthAddress) if len(depthAddress) == len(depthLabels) else 0

    writer = tf.python_io.TFRecordWriter(tfFilename)

    for i in range(numOfData):
        colorImage = Image.open(colorAddress[i])
        colorImage = colorImage.resize((IMAGE_WIDTH,IMAGE_HEIGHT),PIL.Image.ANTIALIAS)
        colorImage = np.asarray(colorImage, np.uint8)

        depthImage = Image.open(depthAddress[i])
        depthImage = depthImage.resize((IMAGE_WIDTH,IMAGE_HEIGHT),PIL.Image.ANTIALIAS)
        depthImage = np.asarray(depthImage, np.uint8)
        
        colorRaw = colorImage.tostring()
        depthRaw = depthImage.tostring()
    

        label = colorLabels[i] if colorLabels[i] == depthLabels[i] else NUM_OF_LABELS

        example = tf.train.Example(features=tf.train.Features(feature={
            'label':_int64_feature(label),
            'colorRaw':_bytes_feature(colorRaw),
            'depthRaw':_bytes_feature(depthRaw)
        }))
        writer.write(example.SerializeToString())
        #print(label)
    writer.close()
    print(numOfData)
    return numOfData

def main():
    writeTFrecord(TRAIN_PATH, TRAIN_DATA_NAME)
    writeTFrecord(TEST_PATH, TEST_DATA_NAME)
    

if __name__ == '__main__':
    main()

