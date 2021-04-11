import tensorflow as tf
import kaggle_Face_expression.GAIN_0905.GAIN_DenseNet121 as GAIN
from tensorflow.python.platform import gfile


def initialize_model(sess, saver, train_dir, expect_exists=False):
    print(train_dir)
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    if (ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path))):
        print("Reading model parameters from", ckpt.model_checkpoint_path)

        sess.run(init_op)

        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        if expect_exists:
            raise Exception("There is no saved checkpoint at", train_dir)
        else:
            print("There is no saved checkpoint at", train_dir, ". Creating new model")
        sess.run(init_op)


def input_size_print(net):
    print("***************************************************************")
    print("the input size is " + str(net.input_width) + "by " + str(net.input_height))
    print("***************************************************************")


logdir = "./logdir_resnet101_8"
modelname = "tf-resnet101_8.ckpt"
input_size = [48, 48, 1]
batch_size = 32
num_classes = 6
num_epoch = 1

net = GAIN.GAIN_DenseNet121(_input_size=input_size, _batch_size=batch_size, _num_classes=num_classes,
                            _num_epoch=num_epoch, isTraining=False, logdir=logdir)
input_size_print(net)
with tf.Session() as sess:
    initialize_model(sess, net.saver, logdir)
    net.evaluate(sess, modelname)
