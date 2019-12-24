import tensorflow as tf
import skimage.io as io
import numpy as np

TRAIN_PATH = './Data/Test'
TRAIN_DATA_NAME = 'trainData.tfrecords'

TEST_PATH = './Data/Training'
TEST_DATA_NAME = 'testData.tfrecord'

DATA_LABELS = ['lever', 'push', 'clothes', 'dial', 'sliding','onOff']
NUM_OF_LABELS = 6

NUM_OF_DATA = 1100

#harcoding image sizels
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
COLOR_NUM_OF_CHANNELS = 3
DEPTH_NUM_OF_CHANNELS = 1
COL_IMAGE_SIZE = [IMAGE_WIDTH, IMAGE_HEIGHT, COLOR_NUM_OF_CHANNELS]
COL_IMAGE_SIZE_FLAT = IMAGE_WIDTH *  IMAGE_HEIGHT * COLOR_NUM_OF_CHANNELS

DEP_IMAGE_SIZE = [IMAGE_WIDTH, IMAGE_HEIGHT, DEPTH_NUM_OF_CHANNELS]
DEP_IMAGE_SIZE_FLAT = IMAGE_WIDTH *  IMAGE_HEIGHT * DEPTH_NUM_OF_CHANNELS

#LEARNING PARAMS
BATCH_SIZE = 25
NUM_OF_EPOCHS = 25

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
    
    sparseLabel = tf.cast(features['label'], tf.int64)

    colorImage = tf.decode_raw(features['colorRaw'],tf.uint8)
    colorImage = tf.reshape(colorImage,COL_IMAGE_SIZE)

    depthImage = tf.decode_raw(features['depthRaw'], tf.uint8)
    depthImage = tf.reshape(depthImage, DEP_IMAGE_SIZE)

    slnh = sparseLabel
    sparseLabel = tf.one_hot(
        sparseLabel,
        depth = NUM_OF_LABELS, 
        on_value = 1.0, 
        off_value = 0.0)

    
    """ 
    colorImages, depthImages, sparseLabels = tf.train.shuffle_batch(
        [colorImage, depthImage, sparseLabel],
        batch_size = batchSize,
        capacity=30,
        min_after_dequeue=10
    )
     
    """
    colorImages, depthImages, sparseLabels = tf.train.batch(
        [colorImage, depthImage, sparseLabel],
        batch_size = batchSize
    ) 
    return colorImages, depthImages, sparseLabels, slnh

filename_queue = tf.train.string_input_producer([TEST_DATA_NAME], num_epochs=10)

# Even when reading in multiple threads, share the filename
# queue.
color,depth, label, slnh = readAndDecode(filename_queue,BATCH_SIZE)

argMax= tf.argmax(label, dimension=1)
# The op for initializing the variables.
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

with tf.Session()  as sess:
    
    sess.run(init_op)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
  
    numOfIter = int(NUM_OF_DATA/BATCH_SIZE)
    totalClassCalled = np.zeros(NUM_OF_LABELS)
    # Let's read off 3 batches just for example
    for i in xrange(numOfIter):
    
        lbl,armax, labelNH = sess.run([label,argMax,slnh])
        totalClassCalled += np.sum(lbl, axis=0)

        print(lbl)
        print(armax)
        print(labelNH)
        print('num of data processed {0}'.format((i+1)*BATCH_SIZE))
        print(totalClassCalled)  
        
    coord.request_stop()
    coord.join(threads)