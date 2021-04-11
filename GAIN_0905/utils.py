import skimage
import skimage.io
import skimage.transform
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.image as mpimg

import tensorflow as tf

import cv2, os

from skimage import io
from skimage.transform import resize

from kaggle_Face_expression.GAIN_0905.check_dir import check_dir

# synset = [l.strip() for l in open('synset.txt').readlines()]

input_height = 48
input_width = 48

def resnet_preprocess(resized_inputs):
    """Faster R-CNN Resnet V1 preprocessing.

    VGG style channel mean subtraction as described here:
    https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md

    Args:
      resized_inputs: A [batch, height_in, width_in, channels] float32 tensor
        representing a batch of images with values between 0 and 255.0.

    Returns:
      preprocessed_inputs: A [batch, height_out, width_out, channels] float32
        tensor representing a batch of images.
    """
    channel_means = tf.constant([123.68, 116.779, 103.939],
        dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
    return resized_inputs - channel_means


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path, normalize=True):
    """
    args:
        normalize: set True to get pixel value of 0~1
    """
    # load image
    img = skimage.io.imread(path)
    if normalize:
        img = img / 255.0
        assert (0 <= img).all() and (img <= 1.0).all()


    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224), preserve_range=True) # do not normalize at transform.
    return resized_img

# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print("Top1: ", top1, prob[pred[0]])
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print("Top5: ", top5)
    return top1



def visualize(image, name, conv_output, conv_grad, gb_viz, cnt_, vls, pred):
    output = conv_output           # [7,7,512]
    grads_val = conv_grad          # [7,7,512]
    print("grads_val shape:", grads_val.shape)
    print("gb_viz shape:", gb_viz.shape)

    weights = np.mean(grads_val, axis = (0, 1)) # alpha_k, [512]
    cam = np.zeros(output.shape[0 : 2], dtype = np.float32)	# [7,7]


    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # Passing through ReLU
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam) # scale 0 to 1.0
    cam = resize(cam, (input_width, input_height), preserve_range=True)

    img = image.astype(float)
    img -= np.min(img)
    img /= img.max()
    # print(img)
    cam_heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_RAINBOW)
    cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
    blend = (cam_heatmap*0.2 + image *0.8) /1.0
    # cam = 255 * cam / np.max(cam)
    # cam = np.uint8(cam)

    str_name= str(name)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    cv2.imwrite(os.path.join(check_dir('heat_map_blend'), str_name[2:len(str_name) - 1] +'_pred_'+str(pred)+'.png'),
                blend)

def save_images(x, label, mask_x,  fileName, logits):
    #print(gcam)

    threshold = 0

    # cam_heatmap = cv2.applyColorMap(np.uint8(255*gcam[:,:,0]), cv2.COLORMAP_RAINBOW)
    # cam_heatmap = cv2.applyColorMap(np.uint8(255*gcam), cv2.COLORMAP_RAINBOW)
    # cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)

    x = np.squeeze(np.uint8(x))
    mask = np.squeeze(np.uint8(mask_x))

    # blend = (cam_heatmap*0.2 + x *0.8) /1.0

    if logits[1]  > logits[0]:
        pred = 1
    else:
        pred = 0

    str_name= str(fileName)
    # cv2.imwrite(os.path.join(check_dir('heat_map_blend_new'), str_name[2:len(str_name) - 1] + '_label_'+str(label) +'_pred_'+str(pred)+'.png'),
                # blend)
    cv2.imwrite(os.path.join(check_dir('input'), str_name[2:len(str_name) - 1] + '.png'),
                x)
    cv2.imwrite(os.path.join(check_dir('mask'), str_name[2:len(str_name) - 1] + '_mask'+'.png'),
                mask)


def save_images2(x, label, non_gcam, cal_gcam, mask_x,  fileName, logits):
    #print(gcam)

    threshold = 10

    if label == 1:
        gcam = cal_gcam
    else:
        gcam = non_gcam

    cam_heatmap = cv2.applyColorMap(np.uint8(255*gcam), cv2.COLORMAP_RAINBOW)
    cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)

    x = np.squeeze(np.uint8(x))
    mask_x = np.squeeze(np.uint8(mask_x))

    blend = (cam_heatmap*0.2 + x *0.8) /1.0

    if logits[1]  > logits[0]:
        pred = 1
    else:
        pred = 0

    str_name= str(fileName)
    cv2.imwrite(os.path.join(check_dir('heat_map_blend_new'), str_name[2:len(str_name) - 1] + '_label_'+str(label) +'_pred_'+str(pred)+'.png'),
                blend)

