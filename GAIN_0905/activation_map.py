# Replace vanila relu to guided relu to get guided backpropagation.
import glob, os, cv2
import numpy as np
import kaggle_Face_expression.GAIN_0905.GAIN_DenseNet121 as GAIN
import kaggle_Face_expression.GAIN_0905.utils

import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

from kaggle_Face_expression.GAIN_0905 import utils


@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(grad.get_shape()))

def read_and_decode(tfrecords_filename, batch_size):

    if os.path.exists(tfrecords_filename) is False:
        raise Exception("no such file" + tfrecords_filename)

    filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=2)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'mask_raw': tf.FixedLenFeature([], tf.string),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'fileName': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.cast(image, tf.float32)
    
    mask = tf.decode_raw(features['mask_raw'], tf.uint8)
    mask = tf.cast( mask, tf.float32 )
        
    ny = features['height']
    nx = features['width']

    image = tf.reshape(image, [input_height, input_width, 3])
    mask = tf.reshape(mask, [input_height,input_width,1])
    
    label = tf.cast(features['label'], tf.int32)
    label = tf.one_hot(label, num_classes)
    label = tf.cast(label, tf.float32)

    fileName = tf.cast(features['fileName'], tf.string)
    
    images, labels, fileNames, masks = tf.train.batch([image, label, fileName, mask],
                                            batch_size=batch_size,
                                            capacity=30,
                                            num_threads=2,
                                            # min_after_dequeue=10,
                                    )
    return images, labels, fileNames, masks

def initialize_model(sess,saver,train_dir,expect_exists=False):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""

    #aaa = tf.Print(v2_path)

    #with tf.Session() as sessin:
    #    sessin.run(aaa)


    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    if( ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path))):
        print("Reading model parameters from", ckpt.model_checkpoint_path)

        sess.run( init_op)

        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        if expect_exists:
            raise Exception("There is no saved checkpoint at", train_dir)
        else:
            print("There is no saved checkpoint at",  train_dir, ". Creating new model")
        sess.run( init_op )

slim = tf.contrib.slim
input_height = 512
input_width = 512
input_size = [512,512,3]
num_classes =2
batch_size = 8

val_filename = '../Python/dataset/valset_image_level_0321_onlyNodule.tfrecord'

logdir = './logdir_test2'
logit_vec0 = []
logit_vec1 = []
ytruth = []

with tf.Session() as sess:
    print('here0')
    val_inputs, val_labels, val_names,val_masks = read_and_decode(val_filename, batch_size)
    print('here1')
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    print('here2')
    try:
        step = 0
        cnt = 0
        cnt_ = 0
        
        
        net = GAIN.GAIN_DenseNet121(_input_size=input_size, _batch_size=batch_size, _num_classes=num_classes, _num_epoch=2, isTraining=False, logdir=logdir)
        
        while not coord.should_stop():
            val_input, val_label, val_name, val_mask =sess.run([val_inputs, val_labels, val_names, val_masks])
            # print(val_input)
            # print(val_label)
            vls_ = tf.argmax(val_label, 1)
            vls = sess.run(vls_)

            
            print('here3')
            initialize_model(sess, net.saver, logdir, expect_exists=True)
            val_input = np.reshape(val_input, [batch_size, input_height, input_width, 3])

            prob = net.end_points['predictions']  # after softmax
            # np.save("prob.npy", prob)

            cost = (-1) * tf.reduce_sum(tf.multiply(tf.log(prob), val_label), axis=1)
            # np.save("cost.npy", sess.run(cost))
            print('cost:', cost)
            print('val_label:', val_label)
            val_label_diff = 1 - val_label
            print('val_label_diff:', val_label_diff)
            y_c = tf.reduce_sum(tf.multiply(net.logits, val_label), axis=1)
            # np.save("y_c.npy", sess.run(y_c))

            print('logit shape: ')
            print(net.logits.shape)
            print('y_c:', y_c)

            # Get last convolutional layer gradient for generating gradCAM visualization
            # print('endpoints:', end_points.keys())

            target_conv_layer = net.end_points['densenet121/dense_block4/conv_block16']
            # target_conv_layer = net.target

            print(target_conv_layer.shape)
            # np.save("target.npy", target_conv_layer)
            target_conv_layer_grad = tf.gradients(y_c, target_conv_layer)[0]
            print('test')

            # print('target_conv_layer_grad: ')
            # print(sess.run(target_conv_layer_grad, feed_dict = {net.x: val_input}))

            gb_grad = tf.gradients(cost, net.x)[0]

            ckpt = tf.train.get_checkpoint_state(logdir)
            v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""

            latest_checkpoint = logdir + '/checkpoint'
            ## Optimistic restore.
            reader = tf.train.NewCheckpointReader(ckpt.model_checkpoint_path)
            saved_shapes = reader.get_variable_to_shape_map()
            variables_to_restore = tf.global_variables()
            for var in variables_to_restore:
                if not var.name.split(':')[0] in saved_shapes:
                    print("WARNING. Saved weight not exists in checkpoint. Init var:", var.name)
                else:
                    # print("Load saved weight:", var.name)
                    pass

            var_names = sorted([(var.name, var.name.split(':')[0]) for var in variables_to_restore
                                if var.name.split(':')[0] in saved_shapes])
            restore_vars = []
            with tf.variable_scope('', reuse=True):
                for var_name, saved_var_name in var_names:
                    try:
                        curr_var = tf.get_variable(saved_var_name)
                        var_shape = curr_var.get_shape().as_list()
                        if var_shape == saved_shapes[saved_var_name]:
                            # print("restore var:", saved_var_name)
                            restore_vars.append(curr_var)
                    except ValueError:
                        print("Ignore due to ValueError on getting var:", saved_var_name)
            saver = tf.train.Saver(restore_vars)

            # prob = sess.run(prob)

            # gb_grad_value, target_conv_layer_value, target_conv_layer_grad_value = sess.run([gb_grad, target_conv_layer, target_conv_layer_grad], feed_dict={images: batch_img, labels: prob})
            
            
            val_preds = tf.argmax(net.logits, axis=1)
            val_preds0 = net.logits[:,0]
            val_preds1 = net.logits[:,1]
            target_conv_layer_value, target_conv_layer_grad_value, gb_grad_value, val_pred,val_preds0_val,val_preds1_val = \
                sess.run([target_conv_layer, target_conv_layer_grad, gb_grad, val_preds,val_preds0,val_preds1], feed_dict = {net.x: val_input})

            for i in range(batch_size):
                # print('See visualization of below category')
                # utils.print_prob(batch_label[i], './synset.txt')
                # utils.print_prob(prob, './synset.txt')
                # print('gb_grad_value[i]:', gb_grad_value[i])
                # print('gb_grad_value[i] shape:', gb_grad_value[i].shape)
                print("pred", val_pred[i])
                cv2.imwrite('./etc/val_' + str(cnt_) + '_label_' + str(vls[i]) + '_pred_'+ str(val_pred[i]) + '.png', val_input[i])
                uv = utils.visualize(val_input[i], val_name[i],
                                     target_conv_layer_value[i],
                                     target_conv_layer_grad_value[i],
                                     gb_grad_value[i],
                                     cnt_,
                                     vls[i],
                                     val_pred[i])
                
                
                logit_vec0 = np.concatenate((logit_vec0, val_preds0_val[i]), axis = None)
                logit_vec1 = np.concatenate((logit_vec1, val_preds1_val[i]), axis = None)
                ytruth = np.concatenate((ytruth, vls[i]), axis = None)
                print("uv: ")
                print(uv)
                cnt_+=1
                
            


        
    except tf.errors.OutOfRangeError:
        
        print('Done training for, %d steps.' % (step))
    finally:
        print(logit_vec0)
        print(logit_vec1)
        print(ytruth)
        np.save('./logit0_am',logit_vec0)
        np.save('./logit1_am',logit_vec1)
        np.save('./logits_truth_am',ytruth)
        # When done, ask the threads to stop.
        coord.request_stop()

    coord.join(threads)
