import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import cv2 as cv
import glob
import os

tf.compat.v1.enable_eager_execution()

Image_Size = (48, 48)
Batch_Size = 32
Data_dir = '.\save'
Shuffle_Number = 30000


def from_tfrecord(serialized):
    features = \
        tf.io.parse_single_example(
            serialized=serialized,
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
    # Image = tf.expand_dims(tf.reshape(Image, [1,48,48]), 0)
    Label = tf.decode_raw(features['Label_Raw'], tf.uint8)
    Label = tf.reshape(Label, [7])
    return Image, Label


def Tensor_to_Array(tensor1):
    """이미지 시각화를 위해 Tesor를 배열로 변환"""
    return np.fromstring(tensor1.numpy(), dtype=int)


Test_queue = '.\Test.tfrecords'
Train_queue = '.\Training.tfrecords'
Validation_queue = '.\Validation.tfrecords'

Train_Dataset = tf.data.TFRecordDataset(Train_queue).map(from_tfrecord)
Train_Dataset = Train_Dataset.shuffle(Shuffle_Number)
Train_Dataset = Train_Dataset.batch(Batch_Size)

Validation_Dataset = tf.data.TFRecordDataset(Validation_queue).map(from_tfrecord)
Validation_Dataset.batch(Batch_Size)

Test_Dataset = tf.data.TFRecordDataset(Test_queue).map(from_tfrecord)
Test_Dataset = Test_Dataset.batch(Batch_Size)

for i, j in Train_Dataset.take(1):
    Show_Images = Tensor_to_Array(i)
    Show_Labels = Tensor_to_Array(j)
    break

print(Show_Images)
print(Show_Images.shape)
# print(Show_Labels)
print(Show_Labels.shape)


'시각화'
Labels_Map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
Fig = plt.figure(figsize=(48, 48))
for i, (px, py) in enumerate(zip(Show_Images, Show_Labels)):
    p = Fig.add_subplot(6, 6, i + 1)
    Label_Name = np.where(py == 1)
    p.set_title("{}".format(Labels_Map[int(Label_Name[0])]), color='blue')
    p.imshow(np.reshape(px, (48, 48)))
    p.axis('off')
    if i == 30:
        break
plt.show()
