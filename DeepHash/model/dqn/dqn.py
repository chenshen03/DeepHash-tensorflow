#################################################################################
# Deep Quantization Network for Efficient Image Retrieval                        #
# Authors: Yue Cao, Mingsheng Long, Jianmin Wang, Han Zhu, Qingfu Wen            #
# Contact: caoyue10@gmail.com                                                    #
##################################################################################

import os
import random
import shutil
import time
from datetime import datetime
from math import ceil

import numpy as np
import tensorflow as tf
from sklearn.cluster import MiniBatchKMeans

from architecture import img_alexnet_layers
from evaluation import MAPs_CQ
from .util import Dataset
from loss import cosine_loss, square_quantization_loss


class DQN(object):
    def __init__(self, config):
        # Initialize setting
        print("initializing")
        np.set_printoptions(precision=4)
        self.stage = tf.placeholder_with_default(tf.constant(0), [])
        self.device = '/gpu:' + str(config.gpu)
        self.output_dim = config.output_dim
        self.n_class = config.label_dim

        self.subspace_num = config.n_subspace
        self.subcenter_num = config.n_subcenter
        self.code_batch_size = config.code_batch_size
        self.cq_lambda = config.cq_lambda
        self.max_iter_update_Cb = config.max_iter_update_Cb
        self.max_iter_update_b = config.max_iter_update_b

        self.batch_size = config.batch_size
        self.val_batch_size = config.val_batch_size
        self.max_iter = config.max_iter
        self.img_model = config.img_model
        self.learning_rate = config.learning_rate
        self.learning_rate_decay_factor = config.learning_rate_decay_factor
        self.decay_step = config.decay_step

        self.finetune_all = config.finetune_all

        self.file_name = 'lr{}_cq{}_ss{}_sc{}_d{}_{}'.format(
            self.learning_rate,
            self.cq_lambda,
            self.subspace_num,
            self.subcenter_num,
            self.output_dim,
            config.dataset)
        self.save_dir = os.path.join(
            config.save_dir, self.file_name + '.npy')
        self.codes_file = os.path.join(
            config.save_dir, self.file_name + '_codes.npy')
        self.log_dir = config.log_dir

        # Setup session
        print("launching session")
        configProto = tf.ConfigProto()
        configProto.gpu_options.allow_growth = True
        configProto.allow_soft_placement = True
        self.sess = tf.Session(config=configProto)

        # Create variables and placeholders

        with tf.device(self.device):
            self.img = tf.placeholder(
                tf.float32, [None, 256, 256, 3])
            self.img_label = tf.placeholder(
                tf.float32, [None, self.n_class])

            self.model_weights = config.model_weights 
            self.img_last_layer, self.deep_param_img, self.train_layers, self.train_last_layer = self.load_model()

            self.C = tf.Variable(tf.random_uniform([self.subspace_num * self.subcenter_num, self.output_dim],
                                                   minval=-1, maxval=1, dtype=tf.float32, name='centers'))
            self.deep_param_img['C'] = self.C

            # Centers shared in different modalities (image & text)
            # Binary codes for different modalities (image & text)
            self.img_output_all = tf.placeholder(
                tf.float32, [None, self.output_dim])
            self.img_b_all = tf.placeholder(
                tf.float32, [None, self.subspace_num * self.subcenter_num])

            self.b_img = tf.placeholder(tf.float32, [None, self.subspace_num * self.subcenter_num])
            self.ICM_m = tf.placeholder(tf.int32, [])
            self.ICM_b_m = tf.placeholder(tf.float32, [None, self.subcenter_num])
            self.ICM_b_all = tf.placeholder(tf.float32, [None, self.subcenter_num * self.subspace_num])
            self.ICM_X = tf.placeholder(tf.float32, [self.code_batch_size, self.output_dim])
            self.ICM_C_m = tf.slice(self.C, [self.ICM_m * self.subcenter_num, 0], [self.subcenter_num, self.output_dim])
            self.ICM_X_residual = self.ICM_X - tf.matmul(self.ICM_b_all, self.C) + tf.matmul(self.ICM_b_m, self.ICM_C_m)
            ICM_X_expand = tf.expand_dims(self.ICM_X_residual, 1)
            ICM_C_m_expand = tf.expand_dims(self.ICM_C_m, 0)
            # N*sc*D  *  D*n
            ICM_sum_squares = tf.reduce_sum(tf.square(tf.squeeze(
                tf.subtract(ICM_X_expand, ICM_C_m_expand))), reduction_indices=2)
            ICM_best_centers = tf.argmin(ICM_sum_squares, 1)
            self.ICM_best_centers_one_hot = tf.one_hot(
                ICM_best_centers, self.subcenter_num, dtype=tf.float32)

            self.global_step = tf.Variable(0, trainable=False)
            self.train_op = self.apply_loss_function(self.global_step)
            self.sess.run(tf.global_variables_initializer())
        return

    def load_model(self):
        if self.img_model == 'alexnet':
            img_output = img_alexnet_layers(
                self.img, self.batch_size, self.output_dim,
                self.stage, self.model_weights, val_batch_size=self.val_batch_size)
        else:
            raise Exception('cannot use such CNN model as ' + self.img_model)
        return img_output

    def save_model(self, model_file=None):
        if model_file is None:
            model_file = self.save_dir
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
        mDataset = collections.namedtuple('Dataset', ['output', 'codes', 'label'])  
        database = mDataset(codes['db_features'], codes['db_codes'], codes['db_label'])
        query = mDataset(codes['val_features'], codes['val_codes'], codes['val_label'])
        C = codes['C']
        return database, query, C

    def save_codes(self, database, query, C, codes_file=None):
        if codes_file is None:
            codes_file = self.codes_file
            # model_file = self.model_weights + "_codes.npy"
        codes = {
            'db_features': database.output,
            'db_codes': database.codes,
            'db_label': database.label,
            'val_features': query.output,
            'val_codes': query.codes,
            'val_label': query.label,
            'C': C,
        }
        print("saving codes to %s" % codes_file)
        np.save(codes_file, np.array(codes))
        return

    def apply_loss_function(self, global_step):
        # loss function
        self.cos_loss = cosine_loss(self.img_last_layer, self.img_label)
        self.q_loss = square_quantization_loss(self.img_last_layer, self.b_img, self.C)
        self.q_lambda = tf.Variable(self.cq_lambda, name='cq_lambda')
        self.loss = self.cos_loss + self.q_lambda * self.q_loss

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
        tf.summary.scalar('cosine_loss', self.cos_loss)
        tf.summary.scalar('quantization_loss', self.q_loss)
        tf.summary.scalar('lr', self.lr)
        self.merged = tf.summary.merge_all()

        # weight层的学习率是bias层的学习率2倍
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

    def initial_centers(self, img_output):
        C_init = np.zeros(
            [self.subspace_num * self.subcenter_num, self.output_dim])
        print("#DQN train# initilizing Centers")
        all_output = img_output
        div = int(self.output_dim / self.subspace_num)
        for i in range(self.subspace_num):
            # TODO debug
            try:
                kmeans = MiniBatchKMeans(n_clusters=self.subcenter_num).fit(all_output[:, i * div: (i + 1) * div])
                C_init[i * self.subcenter_num: (i + 1) * self.subcenter_num, i * div: (i + 1) * div] = kmeans.cluster_centers_
                print("step: ", i, " finish")
            except Exception as e:
                print(e)
                import pdb; pdb.set_trace()
                # raise
        return C_init

    def update_centers(self, img_dataset):
        '''
        Optimize:
            self.C = (U * hu^T + V * hv^T) (hu * hu^T + hv * hv^T)^{-1}
            self.C^T = (hu * hu^T + hv * hv^T)^{-1} (hu * U^T + hv * V^T)
            but all the C need to be replace with C^T :
            self.C = (hu * hu^T + hv * hv^T)^{-1} (hu^T * U + hv^T * V)
        '''
        old_C_value = self.sess.run(self.C)

        h = self.img_b_all
        U = self.img_output_all
        smallResidual = tf.constant(
            np.eye(self.subcenter_num * self.subspace_num, dtype=np.float32) * 0.001)
        Uh = tf.matmul(tf.transpose(h), U)
        hh = tf.add(tf.matmul(tf.transpose(h), h), smallResidual)
        compute_centers = tf.matmul(tf.matrix_inverse(hh), Uh)

        update_C = self.C.assign(compute_centers)
        C_value = self.sess.run(update_C, feed_dict={
            self.img_output_all: img_dataset.output,
            self.img_b_all: img_dataset.codes,
        })

        C_sums = np.sum(np.square(C_value), axis=1)
        C_zeros_ids = np.where(C_sums < 1e-8)
        C_value[C_zeros_ids, :] = old_C_value[C_zeros_ids, :]
        self.sess.run(self.C.assign(C_value))

    def update_codes_ICM(self, output, code):
        '''
        Optimize:
            min || output - self.C * codes ||
            min || output - codes * self.C ||
        args:
            output: [n_train, n_output]
            self.C: [n_subspace * n_subcenter, n_output]
                [C_1, C_2, ... C_M]
            codes: [n_train, n_subspace * n_subcenter]
        '''

        code = np.zeros(code.shape)

        for iterate in range(self.max_iter_update_b):

            sub_list = [i for i in range(self.subspace_num)]
            random.shuffle(sub_list)
            for m in sub_list:
                best_centers_one_hot_val = self.sess.run(self.ICM_best_centers_one_hot, feed_dict={
                    self.ICM_b_m: code[:, m * self.subcenter_num: (m + 1) * self.subcenter_num],
                    self.ICM_b_all: code,
                    self.ICM_m: m,
                    self.ICM_X: output,
                })

                code[:, m * self.subcenter_num: (m + 1) *
                     self.subcenter_num] = best_centers_one_hot_val
        return code

    def update_codes_batch(self, dataset, batch_size):
        '''
        update codes in batch size
        '''
        total_batch = int(ceil(dataset.n_samples / batch_size))
        dataset.finish_epoch()

        for i in range(total_batch):
            output_val, code_val = dataset.next_batch_output_codes(batch_size)
            codes_val = self.update_codes_ICM(output_val, code_val)
            dataset.feed_batch_codes(batch_size, codes_val)

    def train(self, img_dataset):
        print("%s #train# start training" % datetime.now())
        epoch = 0
        epoch_iter = int(ceil(img_dataset.n_samples / self.batch_size))

        # tensorboard
        tflog_path = os.path.join(self.log_dir, self.file_name)
        if os.path.exists(tflog_path):
            shutil.rmtree(tflog_path)
        train_writer = tf.summary.FileWriter(tflog_path, self.sess.graph)

        for train_iter in range(self.max_iter):
            images, labels, codes = img_dataset.next_batch(self.batch_size)
            start_time = time.time()

            if epoch > 0:
                assign_lambda = self.q_lambda.assign(self.cq_lambda)
            else:
                assign_lambda = self.q_lambda.assign(0.0)
            self.sess.run([assign_lambda])

            _, loss, output, summary = self.sess.run([self.train_op, self.loss, self.img_last_layer, self.merged],
                                                     feed_dict={self.img: images,
                                                                self.img_label: labels,
                                                                self.b_img: codes})

            train_writer.add_summary(summary, train_iter)

            img_dataset.feed_batch_output(self.batch_size, output)
            duration = time.time() - start_time

            # every epoch: update codes and centers
            if train_iter % (2 * epoch_iter) == 0 and train_iter != 0:
                epoch = epoch + 1
                for i in range(self.max_iter_update_Cb):
                    self.sess.run(self.C.assign(
                        self.initial_centers(img_dataset.output)))
                    self.update_codes_batch(img_dataset, self.code_batch_size)
            if train_iter % 50 == 0:
                print("%s #train# epoch %2d step %4d, loss = %.4f, %.1f sec/batch"
                      % (datetime.now(), epoch, train_iter + 1, loss, duration))

        print("%s #traing# finish training" % datetime.now())
        self.save_model()
        print("model saved")

        self.sess.close()

    def validation(self, img_database, img_query, R=100):
        if os.path.exists(self.codes_file):
        # if False:
            print("loading ", self.codes_file)
            img_database, img_query, C_tmp = self.load_codes(self.codes_file)
        else:
            print("%s #validation# start validation" % (datetime.now()))
            query_batch = int(ceil(img_query.n_samples / self.val_batch_size))
            print("%s #validation# totally %d query in %d batches" %
                (datetime.now(), img_query.n_samples, query_batch))
            for i in range(query_batch):
                images, labels, codes = img_query.next_batch(self.val_batch_size)
                output, loss = self.sess.run([self.img_last_layer, self.cos_loss],
                                            feed_dict={self.img: images, self.img_label: labels, self.stage: 1})
                img_query.feed_batch_output(self.val_batch_size, output)
                print('Cosine Loss: %s' % loss)

            database_batch = int(ceil(img_database.n_samples / self.val_batch_size))
            print("%s #validation# totally %d database in %d batches" %
                (datetime.now(), img_database.n_samples, database_batch))
            for i in range(database_batch):
                images, labels, codes = img_database.next_batch(self.val_batch_size)

                output, loss = self.sess.run([self.img_last_layer, self.cos_loss],
                                            feed_dict={self.img: images, self.img_label: labels, self.stage: 1})
                img_database.feed_batch_output(self.val_batch_size, output)
                # print output[:10, :10]
                if i % 100 == 0:
                    print('Cosine Loss[%d/%d]: %s' % (i, database_batch, loss))

            self.update_codes_batch(img_query, self.code_batch_size)
            self.update_codes_batch(img_database, self.code_batch_size)

            print("%s #validation# calculating MAP@%d" % (datetime.now(), R))
            C_tmp = self.sess.run(self.C)
            # save features and codes
            self.save_codes(img_database, img_query, C_tmp)

        mAPs = MAPs_CQ(C_tmp, self.subspace_num, self.subcenter_num, R)

        self.sess.close()
        return {
            'i2i_nocq': mAPs.get_mAPs_by_feature(img_database, img_query),
            'i2i_AQD': mAPs.get_mAPs_AQD(img_database, img_query),
            'i2i_SQD': mAPs.get_mAPs_SQD(img_database, img_query)
        }


def train(train_img, config):
    model = DQN(config)
    img_dataset = Dataset(
        train_img, config.output_dim, config.n_subspace * config.n_subcenter)
    model.train(img_dataset)
    return model.save_dir


def validation(database_img, query_img, config):
    model = DQN(config)
    img_database = Dataset(
        database_img, config.output_dim, config.n_subspace * config.n_subcenter)
    img_query = Dataset(
        query_img, config.output_dim, config.n_subspace * config.n_subcenter)
    return model.validation(img_database, img_query, config.R)
