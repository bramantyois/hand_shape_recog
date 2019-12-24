import tensorflow as tf 
import numpy as np 
import time

import fnmatch
import os
import matplotlib.pyplot as plt

import PIL
from PIL import Image

from scipy.misc import toimage
#import skimage.io as io
###################################################################################

TRAIN_DATA_NAME = '../../testData.tfrecord'
TRAIN_DATA_NAME_NOEXT = '../../trainData'

DATA_LABELS = ['sliding', 'onoff', 'pushing', 'cloth', 'rotating','turning','pre']
NUM_OF_LABELS = 7

NUM_OF_DATA = 1149
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
CONV1_FILTER_SIZE = 5
CONV1_NUM_OF_FILTER_COL = 12
CONV1_NUM_OF_FILTER_DEP = 4
CONV1_STRIDE = 1

NORM1_RADIUS = 2
NORM1_ALPHA = 1e-5
NORM1_BETA = 0.75

POOL1_SIZE = [3, 3]
POOL1_STRIDE =2

CONV2_FILTER_SIZE = 5
CONV2_NUM_OF_FILTER_COL = 24
CONV2_NUM_OF_FILTER_DEP = 8
CONV2_STRIDE = 1

NORM2_RADIUS = 2
NORM2_ALPHA = 1e-5
NORM2_BETA = 0.75

POOL2_SIZE = [3, 3]
POOL2_STRIDE = 2

DENSE3_NUM_COL = 192
DENSE3_NUM_DEP = 64
DROP3_PROB = 0.5

DENSE4_NUM = 256

#PARAMETER FOR ITERATION/EPOCH
BATCH_SIZE = 1
LEARNING_RATE = 0.001
NUM_OF_MAX_EPOCH = 200

MIN_ERROR = 1e-12
MODEL_FILENAME = "./TWO_STREAM_MODEL_V3"

###################################################################################
#some prodecures
def normalize(image):
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    return image

def denormalize(image):
    image = (image + 0.5) * 255
    image = image.astype(int)
    return image    

def readAndDecode(filenameQueue, batchSize):
    reader = tf.TFRecordReader()
    _, serializedExample = reader.read(filenameQueue)
    features = tf.parse_single_example(
        serializedExample,
        features={
            'colorRaw': tf.FixedLenFeature([],tf.string),
            'depthRaw':tf.FixedLenFeature([],tf.string),
        })
    colorImage = tf.decode_raw(features['colorRaw'],tf.uint8)
    colorImage = tf.reshape(colorImage,COL_IMAGE_SIZE)
    colorImage = normalize(colorImage)

    depthImage = tf.decode_raw(features['depthRaw'], tf.uint8)
    depthImage = tf.reshape(depthImage, DEP_IMAGE_SIZE)
    depthImage = normalize(depthImage)

    colorImages, depthImages = tf.train.batch(
        [colorImage, depthImage],
        batch_size = batchSize
    ) 

    return colorImages, depthImages

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
    
    colorPool1 = tf.layers.max_pooling2d(
        inputs=colorLayerConv1, 
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
   
    colorPool2 = tf.layers.max_pooling2d(
        inputs=colorLayerConv2, 
        pool_size=POOL2_SIZE, 
        strides=POOL2_STRIDE,
        name="colorPool2")

    colorFlat2,_ = flattenLayer(colorPool2)
    
    #3
    colorDense3 = tf.layers.dense(
        inputs=colorFlat2, 
        units=DENSE3_NUM_COL, 
        activation=tf.nn.relu,
        name='colorDense3')

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
    
    depthPool1 = tf.layers.max_pooling2d(
        inputs=depthLayerConv1, 
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
    
    depthPool2 = tf.layers.max_pooling2d(
        inputs=depthLayerConv2, 
        pool_size=POOL2_SIZE, 
        strides=POOL2_STRIDE,
        name="depthPool2")

    depthFlat2,_ = flattenLayer(depthPool2)
    
    #3
    depthDense3 = tf.layers.dense(
        inputs=depthFlat2, 
        units=DENSE3_NUM_DEP, 
        activation=tf.nn.relu,
        name='depthDense3')

    #COMBINED STREAM
    comb4 = tf.concat([colorDense3, depthDense3],1)

    combDense4 = tf.layers.dense(
        inputs=comb4, 
        units=DENSE4_NUM, 
        activation=tf.nn.relu,
        name='combDense4')

    combDense5 = tf.layers.dense(
        inputs=combDense4, 
        units=NUM_OF_LABELS, 
        activation=None,
        name='combDense5')
    
    #softmax & cross entropy
    y = tf.nn.softmax(combDense5)

    prediction = tf.argmax(y,1)
    
    #tensorflow run
    #loading image
    filenameQueue = tf.train.string_input_producer([TRAIN_DATA_NAME],num_epochs=NUM_OF_MAX_EPOCH )
    colorBatch, depthBatch =  readAndDecode(filenameQueue,BATCH_SIZE)
    
    initOP = tf.group(
        tf.global_variables_initializer(), 
        tf.local_variables_initializer())
    
    saver = tf.train.Saver()


    with tf.Session() as sess:
   
        sess.run(initOP)
        saver.restore(sess, MODEL_FILENAME)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        totalBatch = int(NUM_OF_DATA/BATCH_SIZE)      
        
        startTime = time.time()

        #for epoch in range(NUM_OF_EPOCHS):    
      
        predictionResult= []
        predictionResultText = []
        
        averageAccuracy = 0
        for i in range(totalBatch):
            batchColor, batchDepth = sess.run([colorBatch,depthBatch])
            
            pred = sess.run(
                prediction,
                feed_dict={xColor:batchColor,xDepth:batchDepth})
            
            predictionResult.append(pred)
            
            #print("filename",filename)
            print(DATA_LABELS[pred[0]])
            predictionResultText.append(DATA_LABELS[pred[0]])
            batchColor =batchColor.reshape(IMAGE_HEIGHT,IMAGE_WIDTH,COLOR_NUM_OF_CHANNELS)
            
            #showing image
            #print (type(batchColor))
            #print (batchColor.size)
            #denomColor = denormalize(batchColor)
            #plt.imshow(denomColor)
            #plt.show()
            #toimage(denomColor).show()
            #im = Image.fromarray(denomColor)
            #im.show()
            #printing result
            #print (type(pred))
            #print("result ",(pred))
            
        totalTime = time.time() - startTime
        print("total time", (totalTime))        
        np.savetxt('result.txt', predictionResult, delimiter=',',fmt='%1.1i') 
        
        #writing text result
        thefile = open('result_text.txt','w')
        for item in predictionResultText:   
            thefile.write("%s\n"%item)
        
       
        coord.request_stop()
        coord.join(threads)

        #writertb.close()

if __name__ == '__main__':
    main()

