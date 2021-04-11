import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2 as cv2
from PIL import Image
import time
from kaggle_Face_expression.GAIN_0905.utils import save_images
# from kaggle_Face_expression.GAIN_0905 import lenet
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

            train_inputs1, train_labels1 = self.read_and_decode(train_filename1, self.training_batch_size)

            self.train_input_list = [train_inputs1]
            self.train_label_list = [train_labels1]

            self.test_inputs, self.test_labels = self.read_and_decode_validation(val_filename,
                                                                                 self.validation_batch_size)
        else:
            val_filename = 'C:/Users/user/PycharmProjects/untitled1/kaggle_Face_expression/Test.tfrecord'
            self.validation_set_size = sum(1 for _ in tf.python_io.tf_record_iterator(val_filename))
            print("validation set = ", self.validation_set_size)

            self.test_inputs, self.test_labels = self.read_and_decode_eval(
                val_filename, self.validation_batch_size)

        self.add_placeholder()
        self.logits = self.add_network(self.x)

        self.add_loss(self.logits, self.y_truth)

        if isTraining == True:
            self.add_gradient()
            tf.summary.image("input", self.x[:1], max_outputs=1)
            # tf.summary.image("gcam", tf.reshape(self.gcam, [1, self.input_height, self.input_width, 1])[:1], max_outputs=1)
            self.summaries = tf.summary.merge_all()

        self.saver = tf.train.Saver(max_to_keep=3)

    def valdiation(self, sess):
        sum = 0.0
        iterations = self.validation_set_size // self.validation_batch_size
        for _ in range(iterations):
            output_feed11 = {
                "truth": self.testtruth,
                "logit": self.testlogits,
                "accuracy": self.accuracy,
            }
            _test_inputs, _test_labels = sess.run([self.test_inputs, self.test_labels])
            # sum = sum + self.accuracy.eval(feed_dict={self.x: \
            #                                               _test_inputs, self.y_truth: _test_labels})
            _results1 = sess.run(output_feed11, feed_dict={self.x: _test_inputs, self.y_truth: _test_labels})
            sum = sum + _results1["accuracy"]
            print("logit", _results1["logit"])
            print("truth", _results1["truth"])
        print("accuracy = ", sum / iterations)

    def mean_image_subtraction(self, image):
        _R_MEAN = 138.96
        _G_MEAN = 138.96
        _B_MEAN = 138.96

        means = [_R_MEAN, _G_MEAN, _B_MEAN]
        stds = [52.06, 52.06, 52.06]
        if image.get_shape().ndims != 4:
            raise ValueError('Input must be of size [batch, height, width, C>0]')
        num_channels = image.get_shape().as_list()[-1]
        if len(means) != num_channels:
            raise ValueError('len(means) must match the number of channels')

        channels = tf.split(axis=3, num_or_size_splits=num_channels, value=image)
        for i in range(num_channels):
            channels[i] = (channels[i] - means[i]) / stds[i]
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
                'image_raw': tf.FixedLenFeature([], tf.string)
            })

        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.cast(image, tf.float32) * (1. / 255)

        ny = features['height']
        nx = features['width']
        print(image)
        image = tf.reshape(image, [self.input_height, self.input_width, 1])

        label = tf.cast(features['label'], tf.int32)
        label = tf.one_hot(label, self.num_classes)
        label = tf.cast(label, tf.float32)

        images, labels = tf.train.shuffle_batch([image, label],
                                                batch_size=batch_size, num_threads=4, capacity=self.training_set_size,
                                                min_after_dequeue=10)
        return images, labels

    def read_and_decode_validation(self, tfrecords_filename, batch_size):

        if os.path.exists(tfrecords_filename) is False:
            raise Exception("no such file" + tfrecords_filename)

        filename_queue = tf.train.string_input_producer([tfrecords_filename])

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'label': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'height': tf.FixedLenFeature([], tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string)
            })

        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.cast(image, tf.float32) * (1. / 255)

        ny = features['height']
        nx = features['width']
        print(image)
        image = tf.reshape(image, [self.input_height, self.input_width, 1])

        label = tf.cast(features['label'], tf.int32)
        label = tf.one_hot(label, self.num_classes)
        label = tf.cast(label, tf.float32)

        images, labels = tf.train.shuffle_batch([image, label],
                                                batch_size=batch_size, num_threads=4, capacity=self.validation_set_size,
                                                min_after_dequeue=10)

        return images, labels

    def add_placeholder(self):
        self.x = tf.placeholder(tf.float32, [None, self.input_height, self.input_width, 1])
        self.y_truth = tf.placeholder(tf.float32, [None, self.num_classes])

    def add_network(self, inputs):
        img = inputs
        # print(img)
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, end_points = resnet_v1.resnet_v1_101(img, num_classes=self.num_classes, is_training=self.isTraining,
                                                      reuse=tf.AUTO_REUSE)

            # logits, end_points = resnet_v1.resnet_v1_50(img, self.num_classes, is_training=self.isTraining)
        # prob = end_points["predictions"]
        return net

    def add_loss(self, logits, truth):
        print("add loss")

        with tf.variable_scope("X-ent"):
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=truth)
            # self.loss = tf.nn.weighted_cross_entropy_with_logits(\logits=logits, targets=truth)
            # print(logit_am, logits)
            # loss_am = tf.reduce_sum(tf.multiply(logit_am, truth), axis=1)

            tf.summary.scalar("loss_cl", tf.reduce_sum(self.loss))
            self.loss_cl = tf.reduce_sum(self.loss)

            self.loss = tf.reduce_sum(self.loss)

            tf.summary.scalar("loss", self.loss)

        with tf.variable_scope("Accuracy"):
            self.testtruth = tf.argmax(truth, 1)
            self.testlogits = tf.argmax(logits, 1)
            prediction = tf.equal(tf.argmax(truth, 1), tf.argmax(logits, 1))
            prediction = tf.cast(prediction, "float32")
            self.accuracy = tf.reduce_mean(prediction)
            tf.summary.scalar("accuracy", self.accuracy)

    def add_gradient(self):
        print("add gradient")
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.global_step = tf.Variable(0, trainable=False)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(0.0006).minimize(self.loss, global_step=self.global_step)

    def train(self, sess, modelname):
        global _results
        print("train")

        writer = tf.summary.FileWriter(self.logdir, sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        output_feed = {
            "train_op": self.train_op,
            "global_step": self.global_step,
            "summaries": self.summaries,
            "loss": self.loss_cl
        }

        try:
            step = 0
            sum_loss = 0
            epoch_value = self.training_set_size // self.training_batch_size
            while not coord.should_stop():
                if step % epoch_value == 1:
                    print("epoch: ", step // epoch_value)
                    print("average_loss: ", sum_loss / epoch_value if step > 1 else sum_loss)
                    save_path = self.saver.save(sess, os.path.join(self.logdir, modelname),
                                                global_step=_results["global_step"])
                    print("saved at", save_path)
                    print(_results["global_step"])
                    sum_loss = 0
                    self.valdiation(sess)

                for train_input, train_label in zip(self.train_input_list, self.train_label_list):
                    _train_inputs, _train_labels = sess.run([train_input, train_label])
                    _results = sess.run(output_feed, feed_dict={self.x: _train_inputs, self.y_truth: _train_labels})

                    sum_loss = sum_loss + _results["loss"] / self.training_batch_size

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
                'image_raw': tf.FixedLenFeature([], tf.string),
            })

        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.cast(image, tf.float32) * (1. / 255)

        ny = features['height']
        nx = features['width']
        print(image)
        image = tf.reshape(image, [self.input_height, self.input_width, 1])

        label = tf.cast(features['label'], tf.int32)
        label = tf.one_hot(label, self.num_classes)
        label = tf.cast(label, tf.float32)

        images, labels = tf.train.batch([image, label],
                                        batch_size=batch_size, num_threads=4, capacity=100)

        return images, labels

    def makeplot(self, inputs, labels, logits, j):
        print("makeplot", inputs.shape)
        _, height, weight, channel = inputs.shape
        Labels_Map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
        inputs = np.array(inputs)
        labels = np.array(labels)
        logits = np.array(logits)

        Fig = plt.figure(j,figsize=(48,48))
        for i, (px, py, pz) in enumerate(zip(inputs, labels, logits)):
            # print("plot", py, pz)
            p = Fig.add_subplot(6, 6, i + 1)
            if py == pz:
                p.set_title("{}".format(Labels_Map[py]), color='blue')
            else:
                p.set_title("{}/{}".format(Labels_Map[py], Labels_Map[pz]), color='red')
            if channel==3:
                p.imshow(px)
            else:
                p.imshow(np.reshape(px, (48, 48)))

            p.axis('off')
            if i == 36:
                break
        plt.show()

    def evaluate(self, sess, modelname):
        print("evaluate")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        output_feed = {
            "loss_cl": self.loss,
            "input": self.x,
            "logits": self.logits,
            "logit": self.testlogits,
            "truth": self.testtruth,
            "accuracy": self.accuracy
        }

        logit_vec0 = []
        logit_vec1 = []
        ytruth = []
        sum = 0.0
        j = 0
        try:
            while not coord.should_stop():

                inputs, labels = sess.run(
                    [self.test_inputs, self.test_labels])
                _results1 = sess.run(output_feed, feed_dict={self.x: inputs, self.y_truth: labels})
                self.makeplot(inputs, _results1["truth"], _results1["logit"], j)
                j = j + 1
                print(_results1["logit"])
                print(_results1["truth"])
                print(_results1["accuracy"])
                sum = sum + _results1["accuracy"]
                vls = np.argmax(labels, 1)
                for i in range(self.training_batch_size):
                    # save_images(_results["input"][i], vls[i],
                    #             _results["logits"][i])

                    logit_vec0 = np.concatenate((logit_vec0, _results1["logits"][i][0]), axis=None)
                    logit_vec1 = np.concatenate((logit_vec1, _results1["logits"][i][1]), axis=None)
                    ytruth = np.concatenate((ytruth, vls[i]), axis=None)



        except tf.errors.OutOfRangeError:
            print('Done eval ')
        finally:
            print(logit_vec0)
            print(logit_vec1)
            print(ytruth)
            print(sum / (self.validation_set_size // self.validation_batch_size))
            np.save('./logit0_dense121_new', logit_vec0)
            np.save('./logit1_dense121_new', logit_vec1)
            np.save('./logits_truth_dense121_new', ytruth)
            # When done, ask the threads to stop.
            coord.request_stop()

        coord.join(threads)

    def application(self, sess, modelname):
        global app_image
        print("application")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        output_feed = {
            "input": self.x,
            "logits": self.logits,
            "logit": self.testlogits,
        }

        logit_vec0 = []
        logit_vec1 = []
        ytruth = []
        sum = 0.0
        j = 0
        try:
            while not coord.should_stop():
                # cam 구동
                capture = cv2.VideoCapture(0)
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
                name = 0
                # 캡쳐 Frame
                while True:
                    ## 윈도우 프레레임에 보임
                    ret, frame = capture.read()
                    w1 = 175
                    w2 = 425
                    h1 = 50
                    h2 = 300
                    cv2.imshow("frame", cv2.rectangle(frame, (w1, h1), (w2, h2), (0, 0, 255), 3))

                    key = cv2.waitKey(33)
                    if key == 26:
                        cv2.imwrite("./application_image/%d.jpg" % (name), frame)
                        print("capture", name)
                        name += 1
                    if key == 27:
                        break

                capture.release()
                cv2.destroyAllWindows()
                app_image_list = []
                gray_image_list = []
                app_label = []
                save_app_folder = "./application_image/"
                for i in os.listdir(save_app_folder):
                    print(i)
                    image = cv2.imread(os.path.join(save_app_folder, i), cv2.IMREAD_COLOR)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    print(image)
                    image = image[h1:h2, w1:w2]
                    gray_image = cv2.imread(os.path.join(save_app_folder, i), cv2.IMREAD_GRAYSCALE)
                    gray_image = gray_image[h1:h2, w1:w2]
                    gray_image = cv2.resize(gray_image, dsize=(48, 48), interpolation=cv2.INTER_AREA)
                    app_image_list.append(image)
                    gray_image_list.append(gray_image)
                    app_label.append([0, 1, 0, 0, 0, 0])
                input_app_image = np.array(app_image_list)
                input_gray_image = np.array(gray_image_list)
                input_app_label = np.array(app_label)
                app_number = input_gray_image.shape[0]
                input_gray_image = np.reshape(input_gray_image, [app_number, 48, 48, 1])
                input_app_image = tf.cast(input_app_image, tf.float32) * (1. / 255)
                input_gray_image = tf.cast(input_gray_image, tf.float32) * (1. / 255)
                app_label = tf.cast(input_app_label, tf.float32)

                inputs, gray_inputs, labels = sess.run(
                    [input_app_image, input_gray_image, app_label])
                _results1 = sess.run(output_feed, feed_dict={self.x: gray_inputs, self.y_truth: labels})
                self.makeplot(inputs, _results1["logit"], _results1["logit"], j)

                print(_results1["logit"])

        except tf.errors.OutOfRangeError:
            print('Done eval ')
        finally:

            # When done, ask the threads to stop.
            coord.request_stop()

        coord.join(threads)
