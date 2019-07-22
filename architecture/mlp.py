import os
import tensorflow as tf
import numpy as np


def txt_mlp_layers(txt, txt_dim, output_dim, stage, model_weights=None, with_tanh=True):
    deep_param_txt = {}
    train_layers = []

    if model_weights is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        model_weights = os.path.join(
            dir_path, "pretrained_model/reference_pretrain.npy")

    net_data = dict(np.load(model_weights, encoding='bytes').item())

    # txt_fc1
    with tf.name_scope('txt_fc1'):
        if 'txt_fc1' not in net_data:
            txt_fc1w = tf.Variable(tf.truncated_normal([txt_dim, 4096],
                                                       dtype=tf.float32,
                                                       stddev=1e-2), name='weights')
            txt_fc1b = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
                                   trainable=True, name='biases')
        else:
            txt_fc1w = tf.Variable(net_data['txt_fc1'][0], name='weights')
            txt_fc1b = tf.Variable(net_data['txt_fc1'][1], name='biases')
        txt_fc1l = tf.nn.bias_add(tf.matmul(txt, txt_fc1w), txt_fc1b)

        txt_fc1 = tf.cond(stage > 0, lambda: tf.nn.relu(
            txt_fc1l), lambda: tf.nn.dropout(tf.nn.relu(txt_fc1l), 0.5))

        train_layers += [txt_fc1w, txt_fc1b]
        deep_param_txt['txt_fc1'] = [txt_fc1w, txt_fc1b]

    # txt_fc2
    with tf.name_scope('txt_fc2'):
        if 'txt_fc2' not in net_data:
            txt_fc2w = tf.Variable(tf.truncated_normal([4096, output_dim],
                                                       dtype=tf.float32,
                                                       stddev=1e-2), name='weights')
            txt_fc2b = tf.Variable(tf.constant(0.0, shape=[output_dim], dtype=tf.float32),
                                   trainable=True, name='biases')
        else:
            txt_fc2w = tf.Variable(net_data['txt_fc2'][0], name='weights')
            txt_fc2b = tf.Variable(net_data['txt_fc2'][1], name='biases')

        txt_fc2l = tf.nn.bias_add(tf.matmul(txt_fc1, txt_fc2w), txt_fc2b)
        if with_tanh:
            txt_fc2 = tf.nn.tanh(txt_fc2l)
        else:
            txt_fc2 = txt_fc2l

        train_layers += [txt_fc2w, txt_fc2b]
        train_layers += [txt_fc2w, txt_fc2b]
        deep_param_txt['txt_fc2'] = [txt_fc2w, txt_fc2b]

    # return the output of text layer
    return txt_fc2, deep_param_txt, train_layers
