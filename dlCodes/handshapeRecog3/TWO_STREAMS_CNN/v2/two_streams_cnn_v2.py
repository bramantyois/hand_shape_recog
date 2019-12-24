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

NUM_OF_DATA = 12902
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
CONV1_FILTER_SIZE = 11
CONV1_NUM_OF_FILTER_COL = 96
CONV1_NUM_OF_FILTER_DEP = 32
CONV1_STRIDE = 4

NORM1_RADIUS = 2
NORM1_ALPHA = 1e-5
NORM1_BETA = 0.75

POOL1_SIZE = [3, 3]
POOL1_STRIDE =2

CONV2_FILTER_SIZE = 5
CONV2_NUM_OF_FILTER_COL = 256
CONV2_NUM_OF_FILTER_DEP = 96
CONV2_STRIDE = 1

NORM2_RADIUS = 2
NORM2_ALPHA = 1e-5
NORM2_BETA = 0.75

POOL2_SIZE = [3, 3]
POOL2_STRIDE = 2

CONV3_FILTER_SIZE = 3
CONV3_NUM_OF_FILTER_COL = 384
CONV3_NUM_OF_FILTER_DEP = 128
CONV3_STRIDE = 1 

CONV4_FILTER_SIZE = 3
CONV4_NUM_OF_FILTER_COL = 384
CONV4_NUM_OF_FILTER_DEP = 128
CONV4_STRIDE = 1 

CONV5_FILTER_SIZE = 3
CONV5_NUM_OF_FILTER_COL = 256
CONV5_NUM_OF_FILTER_DEP = 96
CONV5_STRIDE = 1 

POOL5_SIZE = [3, 3]
POOL5_STRIDE = 2

DENSE6_NUM_COL = 4096
DENSE6_NUM_DEP = 1024

DROP6_PROB = 0.5

DENSE7_NUM_COL = 4096
DENSE7_NUM_DEP = 1024

DROP7_PROB = 0.5

DENSE8_NUM = 5030

#PARAMETER FOR ITERATION/EPOCH
BATCH_SIZE = 50
LEARNING_RATE = 0.001
NUM_OF_MAX_EPOCH = 10

MIN_ERROR = 1e-12
MODEL_FILENAME = "./TWO_STREAM_MODEL_V2"

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
    
    xColor = tf.placeholder(tf.float32, shape=[None,  IMAGE_WIDTH, IMAGE_HEIGHT, COLOR_NUM_OF_CHANNELS], name='xColor')
    xDepth = tf.placeholder(tf.float32, shape=[None,  IMAGE_WIDTH, IMAGE_HEIGHT, DEPTH_NUM_OF_CHANNELS], name='xDepth')
    y_ = tf.placeholder(tf.float32, shape=[None, NUM_OF_LABELS])
   
    #COLOR STREAM
    #1
    colorLayerConv1 = tf.layers.conv2d(
        inputs = xColor,
        filters = CONV1_NUM_OF_FILTER_COL,
        kernel_size=[CONV1_FILTER_SIZE,CONV1_FILTER_SIZE],
        padding="same",
        strides=(CONV1_STRIDE,CONV1_STRIDE),
        activation=tf.nn.relu,
        name="colorLayerConv1")
    
    colorNorm1 = tf.nn.local_response_normalization(
        input=colorLayerConv1,
        depth_radius=NORM1_RADIUS,
        alpha=NORM1_ALPHA,
        beta=NORM1_BETA,
        name="colorNorm1"
    )

    colorPool1 = tf.layers.max_pooling2d(
        inputs=colorNorm1, 
        pool_size=POOL1_SIZE, 
        strides=POOL1_STRIDE,
        name="colorPool1")

    #2
    colorLayerConv2 = tf.layers.conv2d(
        inputs = colorPool1,
        filters = CONV2_NUM_OF_FILTER_COL,
        kernel_size=[CONV2_FILTER_SIZE,CONV2_FILTER_SIZE],
        padding="same",
        strides=(CONV2_STRIDE,CONV2_STRIDE),
        activation=tf.nn.relu,
        name="colorLayerConv2")
    
    colorNorm2 = tf.nn.local_response_normalization(
        input=colorLayerConv2,
        depth_radius=NORM2_RADIUS,
        alpha=NORM2_ALPHA,
        beta=NORM2_BETA,
        name="colorNorm2"
    )

    colorPool2 = tf.layers.max_pooling2d(
        inputs=colorNorm2, 
        pool_size=POOL2_SIZE, 
        strides=POOL2_STRIDE,
        name="colorPool2")

    #3
    colorLayerConv3 = tf.layers.conv2d(
        inputs = colorPool2,
        filters = CONV3_NUM_OF_FILTER_COL,
        kernel_size=[CONV3_FILTER_SIZE,CONV3_FILTER_SIZE],
        padding="same",
        strides=(CONV3_STRIDE,CONV3_STRIDE),
        activation=tf.nn.relu,
        name="colorLayerConv3")

    #4
    colorLayerConv4 = tf.layers.conv2d(
        inputs = colorLayerConv3,
        filters = CONV4_NUM_OF_FILTER_COL,
        kernel_size=[CONV4_FILTER_SIZE,CONV4_FILTER_SIZE],
        padding="same",
        strides=(CONV4_STRIDE,CONV4_STRIDE),
        activation=tf.nn.relu,
        name="colorLayerConv4")

    colorLayerConv5 = tf.layers.conv2d(
        inputs = colorLayerConv4,
        filters = CONV5_NUM_OF_FILTER_COL,
        kernel_size=[CONV5_FILTER_SIZE,CONV5_FILTER_SIZE],
        padding="same",
        strides=(CONV5_STRIDE,CONV5_STRIDE),
        activation=tf.nn.relu,
        name="colorLayerConv5")

    colorPool5 = tf.layers.max_pooling2d(
        inputs=colorLayerConv5, 
        pool_size=POOL2_SIZE, 
        strides=POOL5_STRIDE,
        name="colorPool5")

    colorFlat5,_ = flattenLayer(colorPool5)
    
    #6
    colorDense6 = tf.layers.dense(
        inputs=colorFlat5, 
        units=DENSE6_NUM_COL, 
        activation=tf.nn.relu,
        name='colorDense6')

    colorDropout6 = tf.nn.dropout(
        x=colorDense6,
        keep_prob=DROP6_PROB,
        name="colorDropout6"
    )

    #7
    colorDense7 = tf.layers.dense(
        inputs=colorDropout6, 
        units=DENSE7_NUM_COL, 
        activation=tf.nn.relu,
        name='colorDense7')

    colorDropout7 = tf.nn.dropout(
        x=colorDense7,
        keep_prob=DROP7_PROB,
        name="colorDropout7"
    )

    #DEPTH STREAM
    #1
    depthLayerConv1 = tf.layers.conv2d(
        inputs = xDepth,
        filters = CONV1_NUM_OF_FILTER_DEP,
        kernel_size=[CONV1_FILTER_SIZE,CONV1_FILTER_SIZE],
        padding="same",
        strides=(CONV1_STRIDE,CONV1_STRIDE),
        activation=tf.nn.relu,
        name="depthLayerConv1")
    
    depthNorm1 = tf.nn.local_response_normalization(
        input=depthLayerConv1,
        depth_radius=NORM1_RADIUS,
        alpha=NORM1_ALPHA,
        beta=NORM1_BETA,
        name="depthNorm1"
    )

    depthPool1 = tf.layers.max_pooling2d(
        inputs=depthNorm1, 
        pool_size=POOL1_SIZE, 
        strides=POOL1_STRIDE,
        name="depthPool1")

    #2
    depthLayerConv2 = tf.layers.conv2d(
        inputs = depthPool1,
        filters = CONV2_NUM_OF_FILTER_DEP,
        kernel_size=[CONV2_FILTER_SIZE,CONV2_FILTER_SIZE],
        padding="same",
        strides=(CONV2_STRIDE,CONV2_STRIDE),
        activation=tf.nn.relu,
        name="depthLayerConv2")
    
    depthNorm2 = tf.nn.local_response_normalization(
        input=depthLayerConv2,
        depth_radius=NORM2_RADIUS,
        alpha=NORM2_ALPHA,
        beta=NORM2_BETA,
        name="depthNorm2"
    )

    depthPool2 = tf.layers.max_pooling2d(
        inputs=depthNorm2, 
        pool_size=POOL2_SIZE, 
        strides=POOL2_STRIDE,
        name="depthPool2")

    #3
    depthLayerConv3 = tf.layers.conv2d(
        inputs = depthPool2,
        filters = CONV3_NUM_OF_FILTER_DEP,
        kernel_size=[CONV3_FILTER_SIZE,CONV3_FILTER_SIZE],
        padding="same",
        strides=(CONV3_STRIDE,CONV3_STRIDE),
        activation=tf.nn.relu,
        name="depthLayerConv3")

    #4
    depthLayerConv4 = tf.layers.conv2d(
        inputs = depthLayerConv3,
        filters = CONV4_NUM_OF_FILTER_DEP,
        kernel_size=[CONV4_FILTER_SIZE,CONV4_FILTER_SIZE],
        padding="same",
        strides=(CONV4_STRIDE,CONV4_STRIDE),
        activation=tf.nn.relu,
        name="depthLayerConv4")

    depthLayerConv5 = tf.layers.conv2d(
        inputs = depthLayerConv4,
        filters = CONV5_NUM_OF_FILTER_DEP,
        kernel_size=[CONV5_FILTER_SIZE,CONV5_FILTER_SIZE],
        padding="same",
        strides=(CONV5_STRIDE,CONV5_STRIDE),
        activation=tf.nn.relu,
        name="depthLayerConv5")

    depthPool5 = tf.layers.max_pooling2d(
        inputs=depthLayerConv5, 
        pool_size=POOL2_SIZE, 
        strides=POOL5_STRIDE,
        name="depthPool5")

    depthFlat5,_ = flattenLayer(depthPool5)
    
    #6
    depthDense6 = tf.layers.dense(
        inputs=depthFlat5, 
        units=DENSE6_NUM_DEP, 
        activation=tf.nn.relu,
        name='depthDense6')

    depthDropout6 = tf.nn.dropout(
        x=depthDense6,
        keep_prob=DROP6_PROB,
        name="depthDropout6"
    )

    #7
    depthDense7 = tf.layers.dense(
        inputs=depthDropout6, 
        units=DENSE7_NUM_DEP, 
        activation=tf.nn.relu,
        name='depthDense7')

    depthDropout7 = tf.nn.dropout(
        x=depthDense7,
        keep_prob=DROP7_PROB,
        name="depthDropout7"
    )

    #COMBINED STREAM
    comb8 = tf.concat([colorDropout7, depthDropout7],1)

    combDense8 = tf.layers.dense(
        inputs=comb8, 
        units=DENSE8_NUM, 
        activation=tf.nn.relu,
        name='combDense8')

    combDense9 = tf.layers.dense(
        inputs=combDense8, 
        units=NUM_OF_LABELS, 
        activation=None,
        name='combDense9')
    
    #softmax & cross entropy
    y = tf.nn.softmax(combDense9)
     
    crossEntropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate= LEARNING_RATE).minimize(crossEntropy)
    optimizer = tf.train.AdamOptimizer(learning_rate= LEARNING_RATE).minimize(crossEntropy)

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
        #saver.restore(sess, MODEL_FILENAME)
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
                batchColor, batchDepth, batchLbl = sess.run([colorBatch,depthBatch, labelBatch])
                _,cost, acc = sess.run([optimizer,crossEntropy, accuracy],feed_dict={xColor:batchColor,xDepth:batchDepth, y_:batchLbl})
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

