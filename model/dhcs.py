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

from architecture import *
from loss import *
from util import *
from evaluation import *
from data_provider.pairwise import Dataset


class DHCS(object):
    def __init__(self, config):
        # Initialize setting
        print("initializing")
        np.set_printoptions(precision=4)
        self.stage = tf.placeholder_with_default(tf.constant(0), [])
        self.device = '/gpu:' + config.gpus
        self.bit = config.bit
        self.n_class = config.label_dim
        self.q_lambda = config.q_lambda
        self.b_lambda = config.b_lambda
        self.i_lambda = config.i_lambda
        self.alpha = config.alpha
        self.wordvec_dict = config.wordvec_dict

        self.batch_size = config.batch_size
        self.val_batch_size = config.val_batch_size
        self.max_iter = config.max_iter
        self.network = config.network
        self.learning_rate = config.lr
        self.lr_decay_factor = config.lr_decay_factor
        self.decay_step = config.decay_step
        self.finetune_all = config.finetune_all

        self.save_dir = config.save_dir
        self.model_file = os.path.join(self.save_dir, 'network_weights.npy')
        self.codes_file = os.path.join(self.save_dir, 'codes.npy')
        self.tflog_path = os.path.join(self.save_dir, 'tflog')

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
            try:
                self.wordvec = tf.constant(np.loadtxt(self.wordvec_dict), dtype=tf.float32)
            except:
                print(f'{self.wordvec_dict} does not exist!')
                self.wordvec = None

            self.network_weights = config.network_weights
            self.img_last_layer, self.deep_param_img, self.train_layers = self.load_model()

            self.global_step = tf.Variable(0, trainable=False)
            self.train_op = self.apply_loss_function(self.global_step)
            self.sess.run(tf.global_variables_initializer())
            
            if config.debug == True:
                from tensorflow.python import debug as tf_debug
                self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)


    def load_model(self):
        networks = {'alexnet': img_alexnet_layers, 'vgg16': img_vgg16_layers}
        try:
            img_output = networks[self.network](
                    self.img, self.batch_size, self.bit,
                    self.stage, self.network_weights, self.val_batch_size)
        except:
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
        self.S_loss = exp_loss(self.img_last_layer, self.img_label, self.alpha, self.wordvec)
        self.q_loss = quantization_loss(self.img_last_layer, q_type='L2')
        self.b_loss = balance_loss(self.img_last_layer)
        self.i_loss = independence_loss(self.img_last_layer)
        self.loss = self.S_loss + self.q_lambda * self.q_loss + \
                                  self.b_lambda * self.b_loss + \
                                  self.i_lambda * self.i_loss

        # for debug
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('similar_loss', self.S_loss)
        tf.summary.scalar('quantization_loss', self.q_loss)
        tf.summary.scalar('balance_loss', self.b_loss)
        tf.summary.scalar('independence_loss', self.i_loss)
        self.merged = tf.summary.merge_all()

        # Last layer has a 10 times learning rate
        lr = tf.train.exponential_decay(
            self.learning_rate, global_step, self.decay_step, self.lr_decay_factor, staircase=True)
        opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
        grads_and_vars = opt.compute_gradients(self.loss, self.train_layers)

        capped_grads_and_vars = []
        if self.finetune_all:
            for i, grad in enumerate(grads_and_vars[:-2]):
                if i % 2 == 0:
                    capped_grads_and_vars.append((grad[0], grad[1]))
                else:
                    capped_grads_and_vars.append((grad[0]*2, grad[1]))
        capped_grads_and_vars.append((grads_and_vars[-2][0]*10, grads_and_vars[-2][1]))
        capped_grads_and_vars.append((grads_and_vars[-1][0]*20, grads_and_vars[-1][1]))

        return opt.apply_gradients(capped_grads_and_vars,  global_step=global_step)


    def train(self, img_dataset):
        print("%s #train# start training" % datetime.now())

        # tensorboard
        if os.path.exists(self.tflog_path):
            shutil.rmtree(self.tflog_path)
        train_writer = tf.summary.FileWriter(self.tflog_path, self.sess.graph)

        for train_iter in range(self.max_iter):
            images, labels = img_dataset.next_batch(self.batch_size)

            start_time = time.time()

            _, loss, S_loss, q_loss, output, summary = self.sess.run(
                [self.train_op, self.loss, self.S_loss, self.q_loss, self.img_last_layer, self.merged],
                feed_dict={self.img: images,
                           self.img_label: labels})

            img_dataset.feed_batch_output(self.batch_size, output)
            duration = time.time() - start_time

            train_writer.add_summary(summary, train_iter)
            if train_iter % 100 == 0:
                print("%s #train# step %4d, loss = %.4f, similar loss = %.4f, quantization loss = %.4f, %.1f sec/batch"
                      % (datetime.now(), train_iter + 1, loss, S_loss, q_loss, duration))

        print("%s #traing# finish training" % datetime.now())
        self.save_model()
        print("model saved")

        self.sess.close()


    def validation(self, img_database, img_query, R=100):
        if os.path.exists(self.codes_file):
            print("loading ", self.codes_file)
            img_database, img_query = self.load_codes(self.codes_file)
        else:
            print("%s #validation# start validation" % (datetime.now()))
            query_batch = int(ceil(img_query.n_samples / self.val_batch_size))
            print("%s #validation# totally %d query in %d batches" % (datetime.now(), img_query.n_samples, query_batch))
            for i in range(query_batch):
                images, labels = img_query.next_batch(self.val_batch_size)
                output, loss = self.sess.run([self.img_last_layer, self.S_loss],
                                            feed_dict={self.img: images, self.img_label: labels, self.stage: 1})
                img_query.feed_batch_output(self.val_batch_size, output)
                print('Cosine Loss: %s' % loss)

            database_batch = int(ceil(img_database.n_samples / self.val_batch_size))
            print("%s #validation# totally %d database in %d batches" %
                (datetime.now(), img_database.n_samples, database_batch))
            for i in range(database_batch):
                images, labels = img_database.next_batch(self.val_batch_size)

                output, loss = self.sess.run([self.img_last_layer, self.S_loss],
                                            feed_dict={self.img: images, self.img_label: labels, self.stage: 1})
                img_database.feed_batch_output(self.val_batch_size, output)
                # print output[:10, :10]
                if i % 100 == 0:
                    print('Cosine Loss[%d/%d]: %s' % (i, database_batch, loss))
            # save features and codes
            self.save_codes(img_database, img_query)

        self.sess.close()

        db_feats = img_database.output
        db_codes = sign(img_database.output)
        db_labels = img_database.label
        q_feats = img_query.output
        q_codes = sign(img_query.output)
        q_labels = img_query.label

        print("visualizing data ...")
        plot_tsne(np.row_stack((db_codes, q_codes)), np.row_stack((db_labels, q_labels)), self.save_dir)
        plot_distance(db_feats, db_labels, q_feats, q_labels, self.save_dir)
        print(plot_distribution(db_feats, self.save_dir))

        print("calculating metrics ...")
        mAPs = MAPs(R)
        prec, rec, mmap = mAPs.get_precision_recall_by_Hamming_Radius(img_database, img_query, 2)
        return {
            'mAP_sign': mAPs.get_mAPs_after_sign(img_database, img_query),
            'mAP_WhRank': get_whrank_mAP(q_feats, q_codes, q_labels, db_feats, db_codes, db_labels, Rs=R),
            'mAP_finetune': get_finetune_mAP(q_feats, q_codes, q_labels, db_feats, db_codes, db_labels, Rs=R),
            'mAP_feat': mAPs.get_mAPs_by_feature(img_database, img_query), 
            'RAMAP': get_RAMAP(q_codes, q_labels, db_codes, db_labels),
            'mAP_radius2': mmap,
            'prec_radius2': prec,
            'recall_radius2': rec
        }


def train(train_img, config):
    model = DHCS(config)
    img_dataset = Dataset(train_img, config.bit)
    model.train(img_dataset)
    return model.model_file


def validation(database_img, query_img, config):
    model = DHCS(config)
    img_database = Dataset(database_img, config.bit)
    img_query = Dataset(query_img, config.bit)
    return model.validation(img_database, img_query, config.R)
