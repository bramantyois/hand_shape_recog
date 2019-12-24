import tensorflow as tf 
import numpy as np 
import time

import fnmatch
import os
import matplotlib.pyplot as plt

import PIL
from PIL import Image

import skimage.io as io
###################################################################################
TRAIN_DATA_NAME = '../trainData.tfrecords'
TRAIN_DATA_NAME_NOEXT = '../trainData'

DATA_LABELS = ['lever', 'push', 'clothes', 'dial', 'sliding','onOff']
NUM_OF_LABELS = 6

NUM_OF_DATA = 1153
#harcoding image sizels
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 120
COLOR_NUM_OF_CHANNELS = 3
DEPTH_NUM_OF_CHANNELS = 1
COL_IMAGE_SIZE = [IMAGE_WIDTH, IMAGE_HEIGHT, COLOR_NUM_OF_CHANNELS]
COL_IMAGE_SIZE_FLAT = IMAGE_WIDTH *  IMAGE_HEIGHT * COLOR_NUM_OF_CHANNELS

DEP_IMAGE_SIZE = [IMAGE_WIDTH, IMAGE_HEIGHT, DEPTH_NUM_OF_CHANNELS]
DEP_IMAGE_SIZE_FLAT = IMAGE_WIDTH *  IMAGE_HEIGHT * DEPTH_NUM_OF_CHANNELS

#parameter for convolution
FILTER_SIZE1 = 5
NUM_OF_FILTER1 = 16

FILTER_SIZE2 = 5
NUM_OF_FILTER2 = 96



CONNECTED_LAYER_SIZE = 128

COMBINED_CONNECTED_LAYER_SIZE = 128

#PARAMETER FOR ITERATION/EPOCH
BATCH_SIZE = 50
LEARNING_RATE = 0.001
NUM_OF_MAX_EPOCH = 2000

MIN_ERROR = 1e-12
MODEL_FILENAME = "/tmp/model.ckpt"
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

#####################################################################################
#####################################################################################

#some helper functions
def newWeights(shape,name):
    return tf.Variable(tf.truncated_normal(shape, stddev = 0.05), name=name)

def newBiases(length,name):
    return tf.Variable(tf.constant(0.05, shape=[length]), name=name) 

def newConvLayer(
    input,
    numOfInputChannels,
    filterSize, 
    numOfFilters,
    name,
    usePooling = True):

    shape = [filterSize, filterSize, numOfInputChannels, numOfFilters]

    wName = name+"W"
    bName = name+"B"

    weights = newWeights(shape, name=wName)
    biases = newBiases(length=numOfFilters, name=bName)

    layer = tf.nn.conv2d(   
        input=input,
        filter=weights,
        strides=[1,1,1,1],
        padding='SAME')

    layer = layer + biases

    if usePooling:
        layer = tf.nn.max_pool(
            value=layer,
            ksize=[1,2,2,1],
            strides=[1,2,2,1],
            padding='SAME'
        )

    layer = tf.nn.relu(layer)

    return layer, weights

def flattenLayer(layer):
    layerShape = layer.get_shape()

    numOfFeatures = layerShape[1:4].num_elements()

    layerFlat = tf.reshape(layer,[-1, numOfFeatures])

    return layerFlat, numOfFeatures

def newFullyConnectedLayer(
    input,
    numOfInputs,
    numOfOutputs,
    name,
    useRelu=True):

    wName = name+"W"
    bName = name+"B"

    weights = newWeights(shape=[numOfInputs, numOfOutputs], name=wName)
    biases = newBiases(length=numOfOutputs, name=bName)

    layer = tf.matmul(input, weights) +biases

    if useRelu:
        layer = tf.nn.relu(layer)

    return layer 

#####################################################################################
#####################################################################################

def main():
    
    xColor = tf.placeholder(tf.float32, shape=[None,  IMAGE_WIDTH, IMAGE_HEIGHT, COLOR_NUM_OF_CHANNELS], name='xColor')
    xDepth = tf.placeholder(tf.float32, shape=[None,  IMAGE_WIDTH, IMAGE_HEIGHT, DEPTH_NUM_OF_CHANNELS], name='xDepth')
    y_ = tf.placeholder(tf.float32, shape=[None, NUM_OF_LABELS])
   
    colorLayerConv1, colorWeights1 = newConvLayer(
        input = xColor,
        numOfInputChannels = COLOR_NUM_OF_CHANNELS,
        filterSize = FILTER_SIZE1,
        numOfFilters=NUM_OF_FILTER1,
        name="ColConv1",)
    
    colorLayerConv2, colorWeights2 = newConvLayer(
        input=colorLayerConv1,
        numOfInputChannels=NUM_OF_FILTER1,
        filterSize=FILTER_SIZE2,
        numOfFilters=NUM_OF_FILTER2,        
        name="ColConv2")
    
    colorLayerFlat1, colorNumFeatures1 = flattenLayer(colorLayerConv2)

    colorLayerFlat2 = newFullyConnectedLayer(
        input = colorLayerFlat1,
        numOfInputs = colorNumFeatures1,
        numOfOutputs = CONNECTED_LAYER_SIZE,
        name = "colFlat2")    


    depthLayerConv1, depthWeights1 = newConvLayer(
        input = xDepth,
        numOfInputChannels = DEPTH_NUM_OF_CHANNELS,
        filterSize = FILTER_SIZE1,
        numOfFilters = NUM_OF_FILTER1,
        name="DepConv1")

    depthLayerConv2, depthWeights2 = newConvLayer(
        input = depthLayerConv1,
        numOfInputChannels=NUM_OF_FILTER1,
        filterSize=FILTER_SIZE2,
        numOfFilters=NUM_OF_FILTER2,
        name="DepConv2")

    depthLayerFlat1, depthNumFeatures1 = flattenLayer(depthLayerConv2)
    
    depthLayerFlat2 = newFullyConnectedLayer(
        input = depthLayerFlat1,
        numOfInputs = depthNumFeatures1,
        numOfOutputs = CONNECTED_LAYER_SIZE,
        name = "depFlat2")

    combinedLayerFlat = tf.concat([colorLayerFlat2, depthLayerFlat2],1)
    
    combinedConnectedLayer1 = newFullyConnectedLayer(
        input=combinedLayerFlat,
        numOfInputs= (CONNECTED_LAYER_SIZE + CONNECTED_LAYER_SIZE),
        numOfOutputs=CONNECTED_LAYER_SIZE,
        name="Comb1")
    
    combinedConnectedLayer2 = newFullyConnectedLayer(
        input= combinedConnectedLayer1,
        numOfInputs=CONNECTED_LAYER_SIZE,
        numOfOutputs=NUM_OF_LABELS,
        name="Comb2")
    
    combinedConnectedLayer3 = newFullyConnectedLayer(
        input= combinedConnectedLayer2,
        numOfInputs=NUM_OF_LABELS,
        numOfOutputs=NUM_OF_LABELS,
        name="Comb3")
    
    #softmax & cross entropy
    y = tf.nn.softmax(combinedConnectedLayer3)
     
    crossEntropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
    optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE).minimize(crossEntropy)

    correctPrediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correctPrediction,tf.float32))
    
    #tensorflow run
    #loading image
    filenameQueue = tf.train.string_input_producer([TRAIN_DATA_NAME],num_epochs=NUM_OF_MAX_EPOCH )
    colorBatch, depthBatch, labelBatch =  readAndDecode(filenameQueue,BATCH_SIZE)
    
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
                batchCol, batchDep, batchLbl = sess.run([colorBatch, depthBatch, labelBatch])
                _,cost, acc = sess.run([optimizer,crossEntropy, accuracy],feed_dict={xColor:batchCol, xDepth:batchDep,y_:batchLbl})
                avgCost+=cost/totalBatch
                avgAcc+=acc/totalBatch
            print("Epoch=", (epoch+1)," cost =","{:.3f}".format(avgCost))  
            print("Epoch=", (epoch+1)," accuracy =","{:.3f}".format(avgAcc))  
            diffCost = abs(pastAvgCost - avgCost)
            pastAvgCost = avgCost
            epoch = epoch +1


        saver.save(sess,MODEL_FILENAME)
        totalTime = time.time() - startTime

        totalTime = time.time() - startTime
        print("total time", (totalTime))        
        
        coord.request_stop()
        coord.join(threads)

        #writertb.close()

if __name__ == '__main__':
    main()

