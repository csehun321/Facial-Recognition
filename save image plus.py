import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import cv2 as cv
import os

csv_Data = pd.read_csv('./fer2013.csv')
Image_Size = (48, 48)


def Preprocess(File, Mode):
    """csv를 mode에따라 분류"""
    File = np.array(File)
    File = np.array(File[File[:, 2] == Mode])
    Load_Dataset1 = np.array(File[np.max(File[:, 3:6], axis=1) > 7])
    Load_Dataset2 = np.array(File[np.max(File[:, 7:9], axis=1) > 7])
    Load_Dataset_happy = np.array(File[File[:, 6] > 9])
    Load_Dataset_neutral = np.array(File[File[:, 9] > 9])
    Load_Dataset=np.concatenate((Load_Dataset1,Load_Dataset2,Load_Dataset_happy,Load_Dataset_neutral))
    Dateset_Image = list(map(lambda x: x.split(" "), Load_Dataset[:, 1]))
    Dateset_Image = np.array(Dateset_Image, dtype=np.uint8)
    Dateset_Label = np.argmax(Load_Dataset[:, 3:10], axis=1).astype('uint8')
    return Dateset_Image, Dateset_Label


def SaveImage(Dateset_Image, Dateset_Label, Mode):
    """dataset을 save폴더에 종류별로 저장"""
    File_Name = np.zeros(7)
    j = 0
    for i in Dateset_Image:
        if not os.path.exists(".\save_plus\%s\%d" % (Mode, Dateset_Label[j])):
            os.makedirs(".\save_plus\%s\%d" % (Mode, Dateset_Label[j]))
        cv.imwrite(
            ".\save_plus\%s\%d\%d_%d.jpg" % (Mode, Dateset_Label[j], Dateset_Label[j], File_Name[Dateset_Label[j]]),
            i.reshape(Image_Size))
        File_Name[Dateset_Label[j]] += 1
        j += 1
    print(File_Name)


'csv파일을 불러와 numpy로 변환'
Training_Image, Training_Label = Preprocess(csv_Data, Mode="Training")
Validation_Image, Validation_Label = Preprocess(csv_Data, Mode="PrivateTest")
Test_Image, Test_Label = Preprocess(csv_Data, Mode="PublicTest")

'numpy를 이미지로 변환후 save폴더에 저장'
SaveImage(Training_Image, Training_Label, Mode="Training")
SaveImage(Test_Image, Test_Label, Mode="Test")
SaveImage(Validation_Image, Validation_Label, Mode="Validation")

print(Training_Image.shape)
print(Validation_Image.shape)
print(Test_Image.shape)
Testimage = np.reshape(Training_Image[0], Image_Size)
print(Testimage.shape)

cv.imshow("Test", Testimage)
cv.waitKey(0)
cv.destroyAllWindows()
