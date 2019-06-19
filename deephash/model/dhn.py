##################################################################################
# Deep Hashing Network for Efficient Similarity Retrieval                        #
# Authors: Han Zhu, Mingsheng Long, Jianmin Wang, Yue Cao                        #
# Contact: caoyue10@gmail.com                                                    #
##################################################################################

import os
import shutil
import time
from datetime import datetime
from math import ceil

import numpy as np
import tensorflow as tf

from architecture import img_alexnet_layers
from evaluation import MAPs
from loss import cross_entropy_loss, quantization_loss
from data_provider.pairwise import Dataset


class DHN(object):
    def __init__(self, config):
        # Initialize setting
        print("initializing")
        np.set_printoptions(precision=4)
        self.stage = tf.placeholder_with_default(tf.constant(0), [])
        self.device = '/gpu:' + config.gpu_id
        self.output_dim = config.output_dim
        self.n_class = config.label_dim
        self.cq_lambda = config.cq_lambda
        self.alpha = config.alpha

        self.batch_size = config.batch_size
        self.val_batch_size = config.val_batch_size
        self.max_iter = config.max_iter
        self.network = config.network
        self.learning_rate = config.learning_rate
        self.learning_rate_decay_factor = config.learning_rate_decay_factor
        self.decay_step = config.decay_step

        self.finetune_all = config.finetune_all

        self.model_file = os.path.join(config.save_dir, 'network_weights.npy')
        self.codes_file = os.path.join(config.save_dir, 'codes.npy')
        self.tflog_path = os.path.join(config.save_dir, 'tflog')

        # Setup session
        print("launching session")
        configProto = tf.ConfigProto()
        configProto.gpu_options.allow_growth = True
        configProto.allow_soft_placement = True
        self.sess = tf.Session(config=configProto)

        # Create variables and placeholders

        with tf.device(self.device):
            self.img = tf.placeholder(tf.float32, [None, 256, 256, 3])
            self.img_label = tf.placeholder(tf.float32, [None, self.n_class])

            self.network_weights = config.network_weights
            self.img_last_layer, self.deep_param_img, self.train_layers, self.train_last_layer = self.load_model()

            self.global_step = tf.Variable(0, trainable=False)
            self.train_op = self.apply_loss_function(self.global_step)
            self.sess.run(tf.global_variables_initializer())

            if config.debug == True:
                from tensorflow.python import debug as tf_debug
                self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)

    def load_model(self):
        if self.network == 'alexnet':
            img_output = img_alexnet_layers(
                self.img, self.batch_size, self.output_dim,
                self.stage, self.network_weights, val_batch_size=self.val_batch_size)
        else:
            raise Exception('cannot use such CNN model as ' + self.network)
        return img_output

    def save_model(self, model_file=None):
        if model_file is None:
            model_file = self.model_file
        model = {}
        for layer in self.deep_param_img:
            model[layer] = self.sess.run(self.deep_param_img[layer])
        print("saving model to %s" % model_file)
        folder = os.path.dirname(model_file)
        if os.path.exists(folder) is False:
            os.makedirs(folder)
        np.save(model_file, np.array(model))
        return

    def load_codes(self, codes_file=None):
        if codes_file is None:
            codes_file = self.codes_file
        codes = np.load(codes_file).item()

        import collections
        mDataset = collections.namedtuple('Dataset', ['output', 'label'])  
        database = mDataset(codes['db_features'], codes['db_label'])
        query = mDataset(codes['query_features'], codes['query_label'])
        return database, query

    def save_codes(self, database, query, codes_file=None):
        if codes_file is None:
            codes_file = self.codes_file
        codes = {
            'db_features': database.output,
            'db_label': database.label,
            'query_features': query.output,
            'query_label': query.label,
        }
        print("saving codes to %s" % codes_file)
        np.save(codes_file, np.array(codes))

    def apply_loss_function(self, global_step):
        # loss function
        self.cos_loss = cross_entropy_loss(self.img_last_layer, self.img_label, self.alpha, normed=True, balanced=True)
        self.q_loss = self.cq_lambda * quantization_loss(self.img_last_layer)
        self.loss = self.cos_loss + self.q_loss

        # Last layer has a 10 times learning rate
        self.lr = tf.train.exponential_decay(
            self.learning_rate, global_step, self.decay_step, self.learning_rate_decay_factor, staircase=True)
        opt = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
        grads_and_vars = opt.compute_gradients(
            self.loss, self.train_layers + self.train_last_layer)
        fcgrad, _ = grads_and_vars[-2]
        fbgrad, _ = grads_and_vars[-1]

        # for debug
        self.grads_and_vars = grads_and_vars
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('ce_loss', self.cos_loss)
        tf.summary.scalar('q_loss', self.q_loss)
        tf.summary.scalar('lr', self.lr)
        self.merged = tf.summary.merge_all()

        if self.finetune_all:
            return opt.apply_gradients([(grads_and_vars[0][0], self.train_layers[0]),
                                                         (grads_and_vars[1][0]*2, self.train_layers[1]),
                                                         (grads_and_vars[2][0], self.train_layers[2]),
                                                         (grads_and_vars[3][0]*2, self.train_layers[3]),
                                                         (grads_and_vars[4][0], self.train_layers[4]),
                                                         (grads_and_vars[5][0]*2, self.train_layers[5]),
                                                         (grads_and_vars[6][0], self.train_layers[6]),
                                                         (grads_and_vars[7][0]*2, self.train_layers[7]),
                                                         (grads_and_vars[8][0], self.train_layers[8]),
                                                         (grads_and_vars[9][0]*2, self.train_layers[9]),
                                                         (grads_and_vars[10][0], self.train_layers[10]),
                                                         (grads_and_vars[11][0]*2, self.train_layers[11]),
                                                         (grads_and_vars[12][0], self.train_layers[12]),
                                                         (grads_and_vars[13][0]*2, self.train_layers[13]),
                                                         (fcgrad*10, self.train_last_layer[0]),
                                                         (fbgrad*20, self.train_last_layer[1])],
                                                        global_step=global_step)
        else:
            return opt.apply_gradients([(fcgrad * 10, self.train_last_layer[0]),
                                        (fbgrad * 20, self.train_last_layer[1])], global_step=global_step)

    def train(self, img_dataset):
        print("%s #train# start training" % datetime.now())

        # tensorboard
        if os.path.exists(self.tflog_path):
            shutil.rmtree(self.tflog_path)
        train_writer = tf.summary.FileWriter(self.tflog_path, self.sess.graph)

        for train_iter in range(self.max_iter):
            images, labels = img_dataset.next_batch(self.batch_size)
            start_time = time.time()

            _, loss, cos_loss, q_loss, output, summary = self.sess.run(
                [self.train_op, self.loss, self.cos_loss, self.q_loss, self.img_last_layer, self.merged],
                feed_dict={self.img: images,
                           self.img_label: labels})

            img_dataset.feed_batch_output(self.batch_size, output)
            duration = time.time() - start_time

            if train_iter % 1 == 0:
                train_writer.add_summary(summary, train_iter)
                print("%s #train# step %4d, loss = %.4f, cross_entropy loss = %.4f, quantization loss = %.4f, %.1f sec/batch"
                      % (datetime.now(), train_iter + 1, loss, cos_loss, q_loss, duration))

        print("%s #traing# finish training" % datetime.now())
        self.save_model()
        print("model saved")

        self.sess.close()

    def validation(self, img_query, img_database, R=100):
        if os.path.exists(self.codes_file):
            print("loading ", self.codes_file)
            img_database, img_query = self.load_codes(self.codes_file)
        else:
            print("%s #validation# start validation" % (datetime.now()))
            query_batch = int(ceil(img_query.n_samples / self.val_batch_size))
            print("%s #validation# totally %d query in %d batches" % (datetime.now(), img_query.n_samples, query_batch))
            for i in range(query_batch):
                images, labels = img_query.next_batch(self.val_batch_size)
                output, loss = self.sess.run([self.img_last_layer, self.cos_loss],
                                            feed_dict={self.img: images, self.img_label: labels, self.stage: 1})
                img_query.feed_batch_output(self.val_batch_size, output)
                print('Cosine Loss: %s' % loss)

            database_batch = int(ceil(img_database.n_samples / self.val_batch_size))
            print("%s #validation# totally %d database in %d batches" %
                (datetime.now(), img_database.n_samples, database_batch))
            for i in range(database_batch):
                images, labels = img_database.next_batch(self.val_batch_size)

                output, loss = self.sess.run([self.img_last_layer, self.cos_loss],
                                            feed_dict={self.img: images, self.img_label: labels, self.stage: 1})
                img_database.feed_batch_output(self.val_batch_size, output)
                # print output[:10, :10]
                if i % 100 == 0:
                    print('Cosine Loss[%d/%d]: %s' % (i, database_batch, loss))
            # save features and codes
            self.save_codes(img_database, img_query)

        mAPs = MAPs(R)

        self.sess.close()
        prec, rec, mmap = mAPs.get_precision_recall_by_Hamming_Radius(img_database, img_query, 2)
        return {
            'i2i_by_feature': mAPs.get_mAPs_by_feature(img_database, img_query),
            'i2i_after_sign': mAPs.get_mAPs_after_sign(img_database, img_query),
            'i2i_map_radius_2': mmap,
            'i2i_prec_radius_2': prec,
            'i2i_recall_radius_2': rec
        }


def train(train_img, config):
    model = DHN(config)
    img_dataset = Dataset(train_img, config.output_dim)
    model.train(img_dataset)
    return model.model_file


def validation(database_img, query_img, config):
    model = DHN(config)
    img_database = Dataset(database_img, config.output_dim)
    img_query = Dataset(query_img, config.output_dim)
    return model.validation(img_query, img_database, config.R)
