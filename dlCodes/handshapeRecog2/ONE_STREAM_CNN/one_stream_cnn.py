import tensorflow as tf 
import numpy as np 
import time

import fnmatch
import os
#import matplotlib.pyplot as plt

#import PIL
#from PIL import Image

#import skimage.io as io
###################################################################################

TRAIN_DATA_NAME = '../trainData.tfrecords'
TRAIN_DATA_NAME_NOEXT = '../trainData'

DATA_LABELS = ['sliding', 'onoff', 'pushing', 'cloth', 'rotating','turning','pre']
NUM_OF_LABELS = 7

NUM_OF_DATA = 3954
#harcoding image sizels
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
COLOR_NUM_OF_CHANNELS = 3
DEPTH_NUM_OF_CHANNELS = 1
COL_IMAGE_SIZE = [IMAGE_WIDTH, IMAGE_HEIGHT, COLOR_NUM_OF_CHANNELS]
COL_IMAGE_SIZE_FLAT = IMAGE_WIDTH *  IMAGE_HEIGHT * COLOR_NUM_OF_CHANNELS

DEP_IMAGE_SIZE = [IMAGE_WIDTH, IMAGE_HEIGHT, DEPTH_NUM_OF_CHANNELS]
DEP_IMAGE_SIZE_FLAT = IMAGE_WIDTH *  IMAGE_HEIGHT * DEPTH_NUM_OF_CHANNELS

#parameter for convolution
FILTER_SIZE1 = 5
FILTER_KERNEL_SIZE1 = [FILTER_SIZE1,FILTER_SIZE1]
NUM_OF_FILTER1 = 16

FILTER_SIZE2 = 5
FILTER_KERNEL_SIZE2 = [FILTER_SIZE2,FILTER_SIZE2]
NUM_OF_FILTER2 = 32

FILTER_SIZE3 = 5
FILTER_KERNEL_SIZE3 = [FILTER_SIZE3,FILTER_SIZE3]
NUM_OF_FILTER3 = 64

FILTER_SIZE4 = 5
FILTER_KERNEL_SIZE4 = [FILTER_SIZE4,FILTER_SIZE4]
NUM_OF_FILTER4 = 128

FILTER_SIZE5 = 5
FILTER_KERNEL_SIZE5 = [FILTER_SIZE5,FILTER_SIZE5]
NUM_OF_FILTER5 = 256

CONNECTED_LAYER_SIZE1 = 1024
CONNECTED_LAYER_SIZE2 = 1024
CONNECTED_LAYER_SIZE3 = 512
CONNECTED_LAYER_SIZE4 = 512
CONNECTED_LAYER_SIZE5 = 128
CONNECTED_LAYER_SIZE6 = 128
CONNECTED_LAYER_SIZE7 = 8
CONNECTED_LAYER_SIZE8 = 8

#PARAMETER FOR ITERATION/EPOCH
BATCH_SIZE = 50
LEARNING_RATE = 0.001
NUM_OF_MAX_EPOCH = 10

MIN_ERROR = 1e-12
MODEL_FILENAME = "./ONE_STREAM_MODEL"

###################################################################################
#some prodecures
def normalize(image):
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    return image

def readAndDecode(filenameQueue, batchSize):
    reader = tf.TFRecordReader()
    _, serializedExample = reader.read(filenameQueue)
    features = tf.parse_single_example(
        serializedExample,
        features={
            'label': tf.FixedLenFeature([],tf.int64),
            'colorRaw': tf.FixedLenFeature([],tf.string),
            'depthRaw':tf.FixedLenFeature([],tf.string),
        })
    colorImage = tf.decode_raw(features['colorRaw'],tf.uint8)
    colorImage = tf.reshape(colorImage,COL_IMAGE_SIZE)
    colorImage = normalize(colorImage)

    depthImage = tf.decode_raw(features['depthRaw'], tf.uint8)
    depthImage = tf.reshape(depthImage, DEP_IMAGE_SIZE)
    depthImage = normalize(depthImage)

    sparseLabel = tf.cast(features['label'], tf.int32)

    sparseLabel = tf.one_hot(
        sparseLabel,
        depth = NUM_OF_LABELS, 
        on_value = 1.0, 
        off_value = 0.0)

    colorImages, depthImages, sparseLabels = tf.train.batch(
        [colorImage, depthImage, sparseLabel],
        batch_size = batchSize
    ) 

    return colorImages, depthImages, sparseLabels

def readDecodeAndStack(filenameQueue, batchSize):
    reader = tf.TFRecordReader()
    _, serializedExample = reader.read(filenameQueue)
    features = tf.parse_single_example(
        serializedExample,
        features={
            'label': tf.FixedLenFeature([],tf.int64),
            'colorRaw': tf.FixedLenFeature([],tf.string),
            'depthRaw':tf.FixedLenFeature([],tf.string),
        })
    colorImage = tf.decode_raw(features['colorRaw'],tf.uint8)
    colorImage = tf.reshape(colorImage,COL_IMAGE_SIZE)
    colorImage = normalize(colorImage)

    depthImage = tf.decode_raw(features['depthRaw'], tf.uint8)
    depthImage = tf.reshape(depthImage, DEP_IMAGE_SIZE)
    depthImage = normalize(depthImage)

    stackedImage = tf.concat([colorImage, depthImage],axis=2)
    
    sparseLabel = tf.cast(features['label'], tf.int32)

    sparseLabel = tf.one_hot(
        sparseLabel,
        depth = NUM_OF_LABELS, 
        on_value = 1.0, 
        off_value = 0.0)

    stackedImages, sparseLabels = tf.train.batch(
        [stackedImage, sparseLabel],
        batch_size = batchSize
    ) 

    return stackedImages, sparseLabels

#####################################################################################
#####################################################################################

def flattenLayer(layer):
    layerShape = layer.get_shape()

    numOfFeatures = layerShape[1:4].num_elements()

    layerFlat = tf.reshape(layer,[-1, numOfFeatures])

    return layerFlat, numOfFeatures

#####################################################################################
#####################################################################################

def main():
    
    xStacked = tf.placeholder(tf.float32, shape=[None,  IMAGE_WIDTH, IMAGE_HEIGHT, COLOR_NUM_OF_CHANNELS + DEPTH_NUM_OF_CHANNELS], name='xStacked')

    y_ = tf.placeholder(tf.float32, shape=[None, NUM_OF_LABELS])
   
    colorLayerConv1 = tf.layers.conv2d(
        inputs = xStacked,
        filters = NUM_OF_FILTER1,
        kernel_size=FILTER_KERNEL_SIZE1,
        padding="same",
        activation=tf.nn.relu,
        name="ColConv1",)
    
    colorPool1 = tf.layers.max_pooling2d(
        inputs=colorLayerConv1, 
        pool_size=[2, 2], 
        strides=2)

    colorLayerConv2=tf.layers.conv2d(
        inputs = colorPool1,
        filters = NUM_OF_FILTER2,
        kernel_size=FILTER_KERNEL_SIZE2,
        padding="same",
        activation=tf.nn.relu,
        name="ColConv2",)
    
    colorPool2 = tf.layers.max_pooling2d(
        inputs=colorLayerConv2, 
        pool_size=[2, 2], 
        strides=2)

    colorLayerConv3=tf.layers.conv2d(
        inputs = colorPool2,
        filters = NUM_OF_FILTER3,
        kernel_size=FILTER_KERNEL_SIZE3,
        padding="same",
        activation=tf.nn.relu,
        name="ColConv3",)

    colorPool3 = tf.layers.max_pooling2d(
        inputs=colorLayerConv3, 
        pool_size=[2, 2], 
        strides=2)

    colorLayerConv4=tf.layers.conv2d(
        inputs = colorPool3,
        filters = NUM_OF_FILTER4,
        kernel_size=FILTER_KERNEL_SIZE4,
        padding="same",
        activation=tf.nn.relu,
        name="ColConv4",)

    colorPool4 = tf.layers.max_pooling2d(
        inputs=colorLayerConv4, 
        pool_size=[2, 2], 
        strides=2)
        
    colorLayerConv5=tf.layers.conv2d(
        inputs = colorPool4,
        filters = NUM_OF_FILTER5,
        kernel_size=FILTER_KERNEL_SIZE5,
        padding="same",
        activation=tf.nn.relu,
        name="ColConv5",)

    colorFlat1, _ = flattenLayer(colorLayerConv5)

    dense1 = tf.layers.dense(
        inputs=colorFlat1, 
        units=CONNECTED_LAYER_SIZE1, 
        activation=tf.nn.relu,
        name='dense1')

    dense2 = tf.layers.dense(
        inputs=dense1, 
        units=CONNECTED_LAYER_SIZE2, 
        activation=tf.nn.relu,
        name='dense2')

    dense3 = tf.layers.dense(
        inputs=dense2, 
        units=CONNECTED_LAYER_SIZE3, 
        activation=tf.nn.relu,
        name='dense3')
    
    dense4 = tf.layers.dense(
        inputs=dense3, 
        units=CONNECTED_LAYER_SIZE4, 
        activation=tf.nn.relu,
        name='dense4')

    dense5 = tf.layers.dense(
        inputs=dense4, 
        units=CONNECTED_LAYER_SIZE5, 
        activation=tf.nn.relu,
        name='dense5')  
    
    dense6 = tf.layers.dense(
        inputs=dense5, 
        units=CONNECTED_LAYER_SIZE6, 
        activation=tf.nn.relu,
        name='dense6')  
    
    dense7 = tf.layers.dense(
        inputs=dense6, 
        units=CONNECTED_LAYER_SIZE7, 
        activation=tf.nn.relu,
        name='dense7')  

    dense8 = tf.layers.dense(
        inputs=dense7, 
        units=NUM_OF_LABELS,
        activation=None,
        name='dense8')
    
    #softmax & cross entropy
    y = tf.nn.softmax(dense8)
     
    crossEntropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate= LEARNING_RATE).minimize(crossEntropy)

    correctPrediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correctPrediction,tf.float32))
    
    #tensorflow run
    #loading image
    filenameQueue = tf.train.string_input_producer([TRAIN_DATA_NAME],num_epochs=NUM_OF_MAX_EPOCH )
    stackedBatch, labelBatch =  readDecodeAndStack(filenameQueue,BATCH_SIZE)
    
    initOP = tf.group(
        tf.global_variables_initializer(), 
        tf.local_variables_initializer())
    
    saver = tf.train.Saver()

    with tf.Session() as sess:
   
        sess.run(initOP)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        totalBatch = int(NUM_OF_DATA/BATCH_SIZE)      
        
        startTime = time.time()

        #for epoch in range(NUM_OF_EPOCHS):    
        epoch = 0
        diffCost = 1   
        pastAvgCost = 1;   
          
        while (diffCost > MIN_ERROR) and(epoch < NUM_OF_MAX_EPOCH) :   
            avgCost = 0
            avgAcc = 0
            for i in range(totalBatch):
                batchStacked, batchLbl = sess.run([stackedBatch, labelBatch])
                _,cost, acc = sess.run([optimizer,crossEntropy, accuracy],feed_dict={xStacked:batchStacked, y_:batchLbl})
                avgCost+=cost/totalBatch
                avgAcc+=acc/totalBatch
            print("Epoch=", (epoch+1)," cost =","{:.3f}".format(avgCost))  
            print("Epoch=", (epoch+1)," accuracy =","{:.3f}".format(avgAcc))  
            diffCost = abs(pastAvgCost - avgCost)
            pastAvgCost = avgCost
            epoch = epoch +1


        saver.save(sess,MODEL_FILENAME)
        totalTime = time.time() - startTime

        print("total time", (totalTime))        
        
        coord.request_stop()
        coord.join(threads)

        #writertb.close()

if __name__ == '__main__':
    main()

