import tensorflow as tf
import numpy as np
import os
import kaggle_Face_expression.GAIN_0905.densenet
from kaggle_Face_expression.GAIN_0905.utils import save_images
from kaggle_Face_expression.GAIN_0905 import modified_resnet_v1 as resnet_v1

slim = tf.contrib.slim


class GAIN_DenseNet121:
    def __init__(self, _input_size, _num_classes, _batch_size, _num_epoch, isTraining=True, logdir=None):
        self.num_classes = _num_classes
        self.input_size = _input_size
        self.input_height = _input_size[0]
        self.input_width = _input_size[1]
        self.training_batch_size = _batch_size
        self.validation_batch_size = _batch_size
        self.num_epoch = _num_epoch
        self.isTraining = isTraining

        if isTraining == True:
            if logdir == None:
                raise Exception("logdir is need when training")
            self.logdir = logdir
            train_filename1 = 'C:/Users/user/PycharmProjects/untitled1/kaggle_Face_expression/Training.tfrecord'
            self.training_set_size = sum(1 for _ in tf.python_io.tf_record_iterator(train_filename1))
            print("training set = ", self.training_set_size)

            val_filename = 'C:/Users/user/PycharmProjects/untitled1/kaggle_Face_expression/Validation.tfrecord'
            self.validation_set_size = sum(1 for _ in tf.python_io.tf_record_iterator(val_filename))
            print("validation set = ", self.validation_set_size)

            # self.training_batch_size = 8
            # self.validation_batch_size = 8

            train_inputs1, train_labels1, train_masks = self.read_and_decode(train_filename1, self.training_batch_size)

            self.train_input_list = [train_inputs1]
            self.train_label_list = [train_labels1]
            self.train_mask_list = [train_masks]

            self.test_inputs, self.test_labels, self.test_masks = self.read_and_decode(val_filename,
                                                                                       self.validation_batch_size)
        else:
            val_filename = 'C:/Users/user/PycharmProjects/untitled1/kaggle_Face_expression/Test.tfrecord'
            self.validation_set_size = sum(1 for _ in tf.python_io.tf_record_iterator(val_filename))
            print("validation set = ", self.validation_set_size)

            self.test_inputs, self.test_labels, self.test_masks, self.test_names = self.read_and_decode_eval(
                val_filename, self.validation_batch_size)

        self.add_placeholder()
        self.logits = self.add_network(self.x)

        self.norm_gcam = self.add_gcam(0)
        self.nodule_gcam = self.add_gcam(1)

        # print("gcam", self.non_gcam, self.cal_gcam)

        self.gcam = tf.concat([self.norm_gcam, self.nodule_gcam], axis=3)

        self.get_act_norm()

        self.pixel_act_map_loss = self.get_attentionMap_loss_l2(self.mask)

        self.add_loss(self.logits, self.pixel_act_map_loss, self.y_truth)

        if isTraining == True:
            self.add_gradient()
            tf.summary.image("input", self.x[:1], max_outputs=1)
            tf.summary.image("gcam", tf.reshape(self.norm_gcam[:1], [1, self.input_height, self.input_width, 1])[:1],
                             max_outputs=1)
            # tf.summary.image("input_masked", self.mask_x[:1], max_outputs=1)
            tf.summary.image("true_mask", self.mask[:1], max_outputs=1)
            self.summaries = tf.summary.merge_all()

        self.saver = tf.train.Saver(max_to_keep=3)

    def valdiation(self, sess):
        sum = 0.0
        iterations = self.validation_set_size // self.validation_batch_size
        for _ in range(iterations):
            _test_inputs, _test_labels, _test_masks = sess.run([self.test_inputs, self.test_labels, self.test_masks])
            sum = sum + self.accuracy.eval(feed_dict={self.x: \
                                                          _test_inputs, self.y_truth: _test_labels,
                                                      self.mask: _test_masks})
        print("accuracy = ", sum / iterations)

    def mean_image_subtraction(self, image):
        _R_MEAN = 107.58
        _G_MEAN = 107.58
        _B_MEAN = 107.58

        means = [_G_MEAN]
        stds = [20.83, 20.83, 20.83]
        if image.get_shape().ndims != 4:
            raise ValueError('Input must be of size [batch, height, width, C>0]')
        num_channels = image.get_shape().as_list()[-1]
        if len(means) != num_channels:
            raise ValueError('len(means) must match the number of channels')

        channels = tf.split(axis=3, num_or_size_splits=num_channels, value=image)
        for i in range(num_channels):
            channels[i] -= means[i] / stds[i]
        return tf.concat(axis=3, values=channels)

    def read_and_decode(self, tfrecords_filename, batch_size):

        if os.path.exists(tfrecords_filename) is False:
            raise Exception("no such file" + tfrecords_filename)

        filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=self.num_epoch)

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
                'image_raw': tf.FixedLenFeature([], tf.string)
            })

        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.cast(image, tf.float32)

        mask = tf.decode_raw(features['mask_raw'], tf.uint8)
        mask = tf.cast(mask, tf.float32)

        ny = features['height']
        nx = features['width']
        print(image, mask)
        image = tf.reshape(image, [self.input_height, self.input_width, 1])
        mask = tf.reshape(mask, [self.input_height, self.input_width, 1])

        label = tf.cast(features['label'], tf.int32)
        label = tf.one_hot(label, self.num_classes)
        label = tf.cast(label, tf.float32)

        images, labels, masks = tf.train.shuffle_batch([image, label, mask],
                                                       batch_size=batch_size, num_threads=4, capacity=100,
                                                       min_after_dequeue=10)

        return images, labels, masks

    def add_placeholder(self):
        self.x = tf.placeholder(tf.float32, [None, self.input_height, self.input_width, 1])
        self.y_truth = tf.placeholder(tf.float32, [None, self.num_classes])
        self.mask = tf.placeholder(tf.float32, [None, self.input_height, self.input_width, 1])

    def add_network(self, inputs):
        img = self.mean_image_subtraction(inputs)
        # print(img)
        with slim.arg_scope(resnet_v1.resnet_arg_scope):
            logits, self.end_points, self.target = resnet_v1.resnet_v1_50(img, self.num_classes,
                                                                          is_training=self.isTraining,
                                                                          reuse=tf.AUTO_REUSE)
            # logits, end_points = modified_resnet_v1.resnet_v1_50(img, self.num_classes, is_training=self.isTraining)
        #    prob = end_points["predictions"]
        return logits

    def add_loss(self, logits, pixel_act_map_loss, truth):
        print("add loss")
        with tf.variable_scope("X-ent"):
            # self.loss = tf.nn.softmax_cross_entropy_with_logits_v2( \
            #     logits=logits, labels=truth)
            self.loss = tf.nn.weighted_cross_entropy_with_logits( \
                logits=logits, targets=truth, pos_weight=10)
            # print(logit_am, logits)
            # loss_am = tf.reduce_sum(tf.multiply(logit_am, truth), axis=1)

            tf.summary.scalar("logit_cl_non", tf.reduce_sum(logits[:, 0]))
            tf.summary.scalar("logit_cl_cal", tf.reduce_sum(logits[:, 1]))

            tf.summary.scalar("loss_cl", tf.reduce_sum(self.loss))
            tf.summary.scalar("loss_e", tf.reduce_sum(self.pixel_act_map_loss))

            self.loss_cl = tf.reduce_sum(self.loss)

            # self.loss_am = tf.reduce_sum(tf.multiply( tf.ones([ self.training_batch_size, 1 ]) - tf.reshape(truth[:,0], [self.training_batch_size, 1] ), tf.multiply(softmax_am, truth)), axis=1)
            # self.loss_am = tf.reduce_sum(tf.multiply(softmax_am, truth), axis=1)

            self.loss_pixel = pixel_act_map_loss

            print("loss_e", self.loss_pixel)

            self.loss = tf.reduce_sum(self.loss) + 10 * tf.reduce_sum(self.loss_pixel)
            # self.loss = tf.reduce_sum( self.loss )  + 10 * tf.reduce_sum(self.loss_am) + 50 * tf.reduce_sum(self.loss_pixel)
            # self.loss = tf.reduce_sum( self.loss )  + tf.reduce_sum(self.loss_am)
            # self.loss = tf.reduce_sum( self.loss )  + 50 * tf.reduce_sum(self.loss_pixel)

            tf.summary.scalar("loss", self.loss)

        with tf.variable_scope("Accuracy"):
            prediction = tf.equal(tf.argmax(truth, 1), tf.argmax(logits, 1))
            prediction = tf.cast(prediction, "float")
            self.accuracy = tf.reduce_mean(prediction)
            tf.summary.scalar("accuracy", self.accuracy)

    def add_gradient(self):
        print("add gradient")
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.global_step = tf.Variable(0, trainable=False)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(1e-6).minimize(self.loss, global_step=self.global_step)
            # self.train_op = tf.train.GradientDescentOptimizer(1e-5).minimize(self.loss,global_step=self.global_step)
            # self.check_op = tf.add_check_numerics_ops()

    def add_gcam(self, flag):
        print("add gcam")

        if flag == 0:
            target_label = tf.concat([tf.ones([self.training_batch_size, 1]), tf.zeros([self.training_batch_size, 6])],
                                     axis=1)
        else:
            target_label = tf.concat([tf.zeros([self.training_batch_size, 6]), tf.ones([self.training_batch_size, 1])],
                                     axis=1)

        y_c = tf.reduce_sum(tf.multiply(tf.sigmoid(self.logits), target_label), axis=1)

        target_conv_layer = self.end_points['densenet121/dense_block4/conv_block16']
        target_conv_layer_grad = tf.gradients(y_c, target_conv_layer)[0]

        for i in range(self.training_batch_size):
            weights = tf.reduce_mean(target_conv_layer_grad[i], axis=(0, 1))
            # print("weights", weights)
            cam = tf.ones(target_conv_layer[i].shape[0:2], dtype=tf.float32)
            # print("cam",cam)

            weights = tf.reshape(weights, (1, 1, -1))
            self.weights = weights

            cam = target_conv_layer[0] * weights
            cam = tf.reduce_sum(cam, axis=2)
            # print("cam",cam)
            self.check_cam = cam
            cam = tf.nn.relu(cam)
            cam = tf.reshape(cam, (cam.shape[0], cam.shape[1], 1))
            # print("cam",cam)
            cam3 = tf.image.resize_images(cam, (self.input_height, self.input_width))
            # print("cam3",cam3)
            # print(tf.expand_dims(cam3, axis = 0))
            if i == 0:
                cam_batch = tf.expand_dims(cam3, axis=0)
                # print("cam_batch", cam_batch)
            else:
                cam_batch = tf.concat([cam_batch, tf.expand_dims(cam3, axis=0)], axis=0)
                # print("cam_batch", cam_batch)

        return cam_batch

    def get_act_map(self, sigma=.5, w=8):

        # print("gcam size", self.gcam)

        for i in range(self.training_batch_size):
            # print("batch_gcam_check", self.gcam[i] )
            # self.check = tf.Variable(0)

            gcam = (self.gcam[i] - tf.reduce_min(self.gcam[i])) / (
                        tf.reduce_max(self.gcam[i]) - tf.reduce_min(self.gcam[i]) + 0.0000000001)

            if i == 0:
                self.gcam_norm = tf.expand_dims(gcam, axis=0)
            else:
                self.gcam_norm = tf.concat([self.gcam_norm, tf.expand_dims(gcam, axis=0)], axis=0)

            if i == 0:
                act_map = tf.expand_dims(tf.squeeze(tf.sigmoid(w * (gcam - sigma))), axis=0)
            else:
                act_map = tf.concat([act_map, tf.expand_dims(tf.squeeze(tf.sigmoid(w * (gcam - sigma))), axis=0)],
                                    axis=0)

        print("act_map", act_map)
        return act_map

    def get_act_norm(self):

        for i in range(self.training_batch_size):

            ##gcam = self.gcam[ i , :, :, tf.argmax(self.y_truth[i]) ]
            gcam = self.norm_gcam[i, :, :]

            batch_gcam = (gcam - tf.reduce_min(gcam)) / (tf.reduce_max(gcam) - tf.reduce_min(gcam) + 0.0000000001)

            if i == 0:
                self.gcam_norm = tf.expand_dims(batch_gcam, axis=0)
            else:
                self.gcam_norm = tf.concat([self.gcam_norm, tf.expand_dims(batch_gcam, axis=0)], axis=0)

    def get_act_map_x(self):
        act_map = self.get_act_map()

        num_channels = self.x.get_shape().as_list()[-1]

        channels = tf.split(axis=3, num_or_size_splits=num_channels, value=self.x)

        act_map = tf.reshape(act_map, tf.shape(channels[0]))

        for i in range(num_channels):
            channels[i] -= act_map * channels[i]

        return tf.concat(axis=3, values=channels)

    def add_stream_am(self, masked_image):
        print("add_am_stream")
        img = self.mean_image_subtraction(masked_image)
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            logits, self.end_points_am, self.target_am = resnet_v1.resnet_v1(img, self.num_classes,
                                                                                is_training=self.isTraining, reuse=True)
            # logits, end_points = modified_resnet_v1.resnet_v1_50(img, self.num_classes, is_training=self.isTraining)
        #    prob = end_points["predictions"]
        return logits

    def get_attentionMap_loss(self, mask):
        print("add_attentionMap_loss")

        for i in range(self.training_batch_size):

            mask_norm = (mask[i] - tf.reduce_min(mask[i])) / (
                        tf.reduce_max(mask[i]) - tf.reduce_min(mask[i]) + 0.000000001)

            mask_true_f = tf.contrib.layers.flatten(mask_norm)
            pred_f = tf.contrib.layers.flatten(self.gcam_norm[i, :, :])

            intersection = tf.reduce_sum(mask_true_f * pred_f)

            dice_coefficient = - (2 * intersection + 1) / (tf.reduce_sum(mask_true_f) + tf.reduce_sum(pred_f) + 1)

            if i == 0:
                self.loss_pix = tf.expand_dims(dice_coefficient, axis=0)
            else:
                self.loss_pix = tf.concat([self.loss_pix, tf.expand_dims(dice_coefficient, axis=0)], axis=0)

        return self.loss_pix

    def get_attentionMap_loss_l2(self, mask):
        print("add_attentionMap_loss_l2")

        for i in range(self.training_batch_size):
            mask_norm = (mask[i] - tf.reduce_min(mask[i])) / (
                        tf.reduce_max(mask[i]) - tf.reduce_min(mask[i]) + 0.000000001)
            self.square = tf.square(self.gcam_norm[i] - mask_norm)

            if i == 0:
                self.loss_pix = tf.expand_dims(tf.reduce_sum(self.square), axis=0)
            else:
                self.loss_pix = tf.concat([self.loss_pix, tf.expand_dims(tf.reduce_sum(self.square), axis=0)], axis=0)

        return self.loss_pix / (256 * 256)

    def train(self, sess, modelname):
        print("train")

        writer = tf.summary.FileWriter(self.logdir, sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        output_feed = {
            "train_op": self.train_op,
            "global_step": self.global_step,
            "summaries": self.summaries,
            "loss_cl": self.loss_cl,
            "loss_pixel": self.loss_pixel,
            # "check_op": self.check_op
        }

        try:
            step = 0
            while not coord.should_stop():
                if step % 100 == 1:
                    self.valdiation(sess)
                    save_path = self.saver.save(sess, os.path.join(self.logdir, modelname),
                                                global_step=_results["global_step"])
                    print("saved at", save_path)

                for train_input, train_label, train_mask in zip(self.train_input_list, self.train_label_list,
                                                                self.train_mask_list):
                    _train_inputs, _train_labels, _train_masks = sess.run([train_input, train_label, train_mask])
                    _results = sess.run(output_feed, feed_dict={self.x: _train_inputs, self.y_truth: _train_labels,
                                                                self.mask: _train_masks})

                    writer.add_summary(_results["summaries"], global_step=_results["global_step"])

                step = step + 1

        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (self.training_batch_size, step))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        coord.join(threads)
        writer.close()

    def read_and_decode_eval(self, tfrecords_filename, batch_size):

        if os.path.exists(tfrecords_filename) is False:
            raise Exception("no such file" + tfrecords_filename)

        filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=self.num_epoch)

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
        mask = tf.cast(mask, tf.float32)

        ny = features['height']
        nx = features['width']
        print(image, mask)
        image = tf.reshape(image, [self.input_height, self.input_width, 1])
        mask = tf.reshape(mask, [self.input_height, self.input_width, 1])

        label = tf.cast(features['label'], tf.int32)
        label = tf.one_hot(label, self.num_classes)
        label = tf.cast(label, tf.float32)

        fileName = tf.cast(features['fileName'], tf.string)

        images, labels, masks, fileNames = tf.train.batch([image, label, mask, fileName],
                                                          batch_size=batch_size, num_threads=4, capacity=100)

        return images, labels, masks, fileNames

    def evaluate(self, sess, modelname):
        print("evaluate")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        output_feed = {
            "loss_cl": self.loss_cl,
            "loss_pixel": self.loss_pixel,
            "input": self.x,
            "grad": self.gcam,
            "weights": self.weights,
            "logits": self.logits
        }

        logit_vec0 = []
        logit_vec1 = []
        ytruth = []
        try:
            while not coord.should_stop():

                inputs, labels, masks, fileNames = sess.run(
                    [self.test_inputs, self.test_labels, self.test_masks, self.test_names])
                _results = sess.run(output_feed, feed_dict={self.x: inputs, self.y_truth: labels, self.mask: masks})
                vls = np.argmax(labels, 1)
                for i in range(self.training_batch_size):
                    save_images(_results["input"][i], vls[i], _results["grad"][i], masks[i], fileNames[i],
                                _results["logits"][i])

                    logit_vec0 = np.concatenate((logit_vec0, _results["logits"][i][0]), axis=None)
                    logit_vec1 = np.concatenate((logit_vec1, _results["logits"][i][1]), axis=None)
                    ytruth = np.concatenate((ytruth, vls[i]), axis=None)

        except tf.errors.OutOfRangeError:
            print('Done eval ')
        finally:
            print(logit_vec0)
            print(logit_vec1)
            print(ytruth)
            np.save('./logit0_dense121_cal_0405', logit_vec0)
            np.save('./logit1_dense121_cal_0405', logit_vec1)
            np.save('./logits_truth_dense121_cal_0405', ytruth)
            # When done, ask the threads to stop.
            coord.request_stop()

        coord.join(threads)
