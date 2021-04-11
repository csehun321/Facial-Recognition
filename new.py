import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import cv2 as cv
import glob
import os

###tf.compat.v1.enable_eager_execution()

Image_Size = (48, 48)
batch_size = 1
Data_dir = '.\save'
Shuffle_Number = 30000

def from_tfrecord(tfrecords_filename, batch_size):

    if os.path.exists(tfrecords_filename) is False:
        raise Exception("no such file" + tfrecords_filename)

    filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=100)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)


    features = \
        tf.io.parse_single_example(
            serialized_example,
            features={
                'Width': tf.io.FixedLenFeature([], tf.int64),
                'Height': tf.io.FixedLenFeature([], tf.int64),
                'Num': tf.io.FixedLenFeature([], tf.int64),
                'Num_Class': tf.io.FixedLenFeature([], tf.int64),
                'Image_Raw': tf.io.FixedLenFeature([], tf.string),
                'Label_Raw': tf.io.FixedLenFeature([], tf.string)
            }
        )
    # num = tf.cast(features['num'], tf.int64)
    # print(num)
    Image = tf.decode_raw(features['Image_Raw'], tf.uint8)

    Image = tf.cast(Image, tf.float32)

    Image = tf.reshape(Image, [48, 48, 1])
    Label = tf.decode_raw(features['Label_Raw'], tf.uint8)
    Label = tf.reshape(Label, [7])

    images, labels = tf.train.shuffle_batch([Image, Label],
                                            batch_size=batch_size,
                                            capacity=30,
                                            num_threads=2,
                                            min_after_dequeue=10)

    return images, labels

tfrecords_filename = "./Training.tfrecords"
images, labels = from_tfrecord(tfrecords_filename, batch_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.data.start_queue_runners(coord=coord, sess=sess)

    try:
        step = 0
        while not coord.should_stop():
            _images, _labels = sess.run([images, labels])

            ny = 1
            nx = 1
            assert batch_size == ny * nx
            _images = _images.reshape(ny, nx, 48, 48, 1)

            _labels = np.argmax(_labels, axis=1)

            cv.namedWindow("wnd")
            cv.imshow("wnd", _images[0][0] / 255)
            print(_labels)
            cv.waitKey(0)
            step = step + 1

    except tf.errors.OutOfRangeError:
        print('Done training for, %d steps.' % (step))
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    coord.join(threads)