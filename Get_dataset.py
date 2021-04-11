import glob
import os

import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import kaggle_Face_expression.GAIN_0905.GAIN_DenseNet121 as GAIN

# tf.compat.v1.enable_eager_execution()

sess = tf.InteractiveSession()
logdir = "./logdir_cal_dense_new"
modelname = "tf-densenet121.ckpt"
Input_size = [48, 48, 1]
batch_size = 32
num_classes = 6
num_epoch = 20
Data_dir = './save_plus'

"""Get_dataset 시작"""


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(Images, Labels, Mode, Filenames):
    Num_Examples = Labels.shape[0]
    if Images.shape[0] != Num_Examples:
        raise ValueError("Images size %d does not match label size %d." %
                         (Images.shape[0], Num_Examples))
    Num, Height, Width, Channel = Images.shape
    print("Num_Examples", Num_Examples)
    filename = os.path.join('./', Mode + '.tfrecord')
    print('Writing', filename)
    ##print('image shape', Images.shape)
    writer = tf.io.TFRecordWriter(filename)
    for index in range(Num_Examples):
        Image_Raw = Images[index].tostring()
        Labels_Raw = Labels[index]
        print(Labels_Raw)
        Filename_raw = Filenames[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'width': _int64_feature(Width),
            'height': _int64_feature(Height),
            'label': _int64_feature(Labels_Raw),
            'mask_raw': _bytes_feature(Image_Raw),
            'fileName': _bytes_feature(Filename_raw),
            'image_raw': _bytes_feature(Image_Raw)}))
        writer.write(example.SerializeToString())


def Get_Dataset(Mode, Data_dir):
    """저장된 폴더안의 이미지들을 불러와 Dataset 형성"""
    path = '%s\%s\*' % (Data_dir, Mode)
    Load_Dataset = glob.glob(path)
    Dataset_Image = []
    Dataset_Label = []
    Dataset_Filename = []
    for i in Load_Dataset:
        Imagelist = os.listdir(i)
        Label_Name = i[-1]
        for j in Imagelist:
            Read_Image = cv.imread(os.path.join(i, j), cv.IMREAD_GRAYSCALE)
            # print(np.max(Read_Image),np.min(Read_Image))
            Read_Image = np.reshape(Read_Image, Input_size)
            Dataset_Filename.append(j[2:-4])
            Dataset_Image.append(Read_Image)
            Dataset_Label.append(Label_Name)
    Dataset_Image = np.array(Dataset_Image).astype('uint8')
    print(Dataset_Image.shape)
    Dataset_Label = np.array(Dataset_Label).astype('uint8')
    print(Dataset_Label)
    Dataset_Filename = np.array(Dataset_Filename).astype('uint16')
    Mode_Dataset = tf.data.Dataset.from_tensor_slices((Dataset_Image, Dataset_Label))
    convert_to(Dataset_Image, Dataset_Label, Mode, Dataset_Filename)
    return Mode_Dataset


Train_Dataset = Get_Dataset(Mode="Training", Data_dir=Data_dir)
# Train_Dataset = Train_Dataset.shuffle(Shuffle_Number)
# Train_Dataset = Train_Dataset.batch(Batch_Size)

Validation_Dataset = Get_Dataset(Mode="Validation", Data_dir=Data_dir)
# Validation_Dataset.batch(Batch_Size)

Test_Dataset = Get_Dataset(Mode="Test", Data_dir=Data_dir)
# Test_Dataset = Test_Dataset.batch(Batch_Size)
# print(Test_Dataset)

"""Get_dataset 끝"""

# 'Matplot'
# fig = plt.figure(figsize=(3, 3))
#
# ax1 = fig.add_subplot(3, 3, 1)
# ax1.set_title("image")
# ax1.imshow(emotion_Image1[5])
# plt.show()


# def Tensor_to_Array(tensor1):
#     """이미지 시각화를 위해 Tesor를 배열로 변환"""
#     return tensor1.numpy()
#
#
# for i, j in Train_Dataset.take(1):
#     Show_Images = Tensor_to_Array(i)
#     Show_Labels = Tensor_to_Array(j)
#     break
# print(Show_Labels)
# print(Show_Labels.shape)
# '시각화'
# Labels_Map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
# Fig = plt.figure(figsize=Image_Size)
# for i, (px, py) in enumerate(zip(Show_Images, Show_Labels)):
#     p = Fig.add_subplot(6, 6, i + 1)
#     Label_Name = np.where(py == 1)
#     p.set_title("{}".format(Labels_Map[int(Label_Name[0])]), color='blue')
#     p.imshow(px)
#     p.axis('off')
# plt.show()
