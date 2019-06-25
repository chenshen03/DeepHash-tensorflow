import os
import tensorflow as tf
import numpy as np


def img_vgg16_layers(img, batch_size, output_dim, stage, model_weights=None, val_batch_size=32, with_tanh=True):
    deep_param_img = {}
    train_layers = []

    if model_weights is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        model_weights = os.path.join(dir_path, "pretrained_model/vgg16_weights.npy")
    
    print("loading img model from ", model_weights)
    net_data = dict(np.load(model_weights, encoding='bytes').item())
    print(list(net_data.keys()))

    # swap(2,1,0), bgr -> rgb
    reshaped_image = tf.cast(img, tf.float32)[:, :, :, ::-1]

    height = 224
    width = 224

    # Randomly crop a [height, width] section of each image
    with tf.name_scope('preprocess'):
        def train_fn():
            return tf.stack([tf.random_crop(tf.image.random_flip_left_right(each), [height, width, 3])
                             for each in tf.unstack(reshaped_image, batch_size)])

        def val_fn():
            unstacked = tf.unstack(reshaped_image, val_batch_size)

            def crop(img, x, y): return tf.image.crop_to_bounding_box(
                img, x, y, width, height)

            def distort(f, x, y): return tf.stack(
                [crop(f(each), x, y) for each in unstacked])

            def distort_raw(x, y): return distort(lambda x: x, x, y)

            def distort_fliped(x, y): return distort(
                tf.image.flip_left_right, x, y)
            distorted = tf.concat([distort_fliped(0, 0), distort_fliped(28, 0),
                                   distort_fliped(
                                       0, 28), distort_fliped(28, 28),
                                   distort_fliped(14, 14), distort_raw(0, 0),
                                   distort_raw(28, 0), distort_raw(0, 28),
                                   distort_raw(28, 28), distort_raw(14, 14)], 0)

            return distorted
        distorted = tf.cond(stage > 0, val_fn, train_fn)

        # Zero-mean input
        mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32, shape=[
                           1, 1, 1, 3], name='img-mean')
        distorted = distorted - mean

    # conv1_1
    with tf.name_scope('conv1_1') as scope:
        kernel = tf.Variable(net_data['conv1_1'][0], name='weights')
        conv = tf.nn.conv2d(distorted, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv1_1'][1], trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1_1 = tf.nn.relu(out, name=scope)
        deep_param_img['conv1_1'] = [kernel, biases]
        train_layers += [kernel, biases]

    # conv1_2
    with tf.name_scope('conv1_2') as scope:
        kernel = tf.Variable(net_data['conv1_2'][0], name='weights')
        conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv1_2'][1],
                                trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1_2 = tf.nn.relu(out, name=scope)
        deep_param_img['conv1_2'] = [kernel, biases]
        train_layers += [kernel, biases]

    # pool1
    pool1 = tf.nn.max_pool(conv1_2,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            name='pool1')

    # conv2_1
    with tf.name_scope('conv2_1') as scope:
        kernel = tf.Variable(net_data['conv2_1'][0], name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv2_1'][1],
                                trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv2_1 = tf.nn.relu(out, name=scope)
        deep_param_img['conv2_1'] = [kernel, biases]
        train_layers += [kernel, biases]

    # conv2_2
    with tf.name_scope('conv2_2') as scope:
        kernel = tf.Variable(net_data['conv2_2'][0], name='weights')
        conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv2_2'][1],
                                trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv2_2 = tf.nn.relu(out, name=scope)
        deep_param_img['conv2_2'] = [kernel, biases]
        train_layers += [kernel, biases]

    # pool2
    pool2 = tf.nn.max_pool(conv2_2,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            name='pool2')

    # conv3_1
    with tf.name_scope('conv3_1') as scope:
        kernel = tf.Variable(net_data['conv3_1'][0], name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv3_1'][1],
                                trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_1 = tf.nn.relu(out, name=scope)
        deep_param_img['conv3_1'] = [kernel, biases]
        train_layers += [kernel, biases]

    # conv3_2
    with tf.name_scope('conv3_2') as scope:
        kernel = tf.Variable(net_data['conv3_2'][0], name='weights')
        conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv3_2'][1],
                                trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_2 = tf.nn.relu(out, name=scope)
        deep_param_img['conv3_2'] = [kernel, biases]
        train_layers += [kernel, biases]

    # conv3_3
    with tf.name_scope('conv3_3') as scope:
        kernel = tf.Variable(net_data['conv3_3'][0], name='weights')
        conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv3_3'][1],
                                trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_3 = tf.nn.relu(out, name=scope)
        deep_param_img['conv3_3'] = [kernel, biases]
        train_layers += [kernel, biases]

    # pool3
    pool3 = tf.nn.max_pool(conv3_3,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            name='pool3')

    # conv4_1
    with tf.name_scope('conv4_1') as scope:
        kernel = tf.Variable(net_data['conv4_1'][0], name='weights')
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv4_1'][1],
                                trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_1 = tf.nn.relu(out, name=scope)
        deep_param_img['conv4_1'] = [kernel, biases]
        train_layers += [kernel, biases]

    # conv4_2
    with tf.name_scope('conv4_2') as scope:
        kernel = tf.Variable(net_data['conv4_2'][0], name='weights')
        conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv4_2'][1],
                                trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_2 = tf.nn.relu(out, name=scope)
        deep_param_img['conv4_2'] = [kernel, biases]
        train_layers += [kernel, biases]

    # conv4_3
    with tf.name_scope('conv4_3') as scope:
        kernel = tf.Variable(net_data['conv4_3'][0], name='weights')
        conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv4_3'][1],
                                trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_3 = tf.nn.relu(out, name=scope)
        deep_param_img['conv4_3'] = [kernel, biases]
        train_layers += [kernel, biases]

    # pool4
    pool4 = tf.nn.max_pool(conv4_3,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            name='pool4')

    # conv5_1
    with tf.name_scope('conv5_1') as scope:
        kernel = tf.Variable(net_data['conv5_1'][0], name='weights')
        conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv5_1'][1],
                                trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_1 = tf.nn.relu(out, name=scope)
        deep_param_img['conv5_1'] = [kernel, biases]
        train_layers += [kernel, biases]

    # conv5_2
    with tf.name_scope('conv5_2') as scope:
        kernel = tf.Variable(net_data['conv5_2'][0], name='weights')
        conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv5_2'][1],
                                trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_2 = tf.nn.relu(out, name=scope)
        deep_param_img['conv5_2'] = [kernel, biases]
        train_layers += [kernel, biases]

    # conv5_3
    with tf.name_scope('conv5_3') as scope:
        kernel = tf.Variable(net_data['conv5_3'][0], name='weights')
        conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv5_3'][1],
                                trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_3 = tf.nn.relu(out, name=scope)
        deep_param_img['conv5_3'] = [kernel, biases]
        train_layers += [kernel, biases]

    # pool5
    pool5 = tf.nn.max_pool(conv5_3,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            name='pool4')

    # fc6
    with tf.name_scope('fc6') as scope:
        shape = int(np.prod(pool5.get_shape()[1:]))
        fc6w = tf.Variable(net_data['fc6'][0], name='weights')
        fc6b = tf.Variable(net_data['fc6'][1],
                                trainable=True, name='biases')
        pool5_flat = tf.reshape(pool5, [-1, shape])
        fc6l = tf.nn.bias_add(tf.matmul(pool5_flat, fc6w), fc6b)
        fc6 = tf.nn.relu(fc6l)
        deep_param_img['fc6'] = [fc6w, fc6b]
        train_layers += [fc6w, fc6b]

    # fc7
    with tf.name_scope('fc7') as scope:
        fc7w = tf.Variable(net_data['fc7'][0], name='weights')
        fc7b = tf.Variable(net_data['fc7'][1],
                                trainable=True, name='biases')
        fc7l = tf.nn.bias_add(tf.matmul(fc6, fc7w), fc7b)
        fc7 = tf.nn.relu(fc7l)
        deep_param_img['fc7'] = [fc7w, fc7b]
        train_layers += [fc7w, fc7b]

    # # fc8
    # with tf.name_scope('fc8') as scope:
    #     fc8w = tf.Variable(tf.truncated_normal([4096, 1000],
    #                                                     dtype=tf.float32,
    #                                       net_data['conv1_1_W'], name='weights')
    #     fc8b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
    #                             trainable=True, name='biases')
    #     fc8l = tf.nn.bias_add(tf.matmul(fc7, fc8w), fc8b)
    #     train_layers += [fc8w, fc8b]

    # FC8
    # Output output_dim
    with tf.name_scope('fc8'):
        # Differ train and val stage by 'fc8' as key
        if 'fc8' in net_data:
            fc8w = tf.Variable(net_data['fc8'][0], name='weights')
            fc8b = tf.Variable(net_data['fc8'][1], name='biases')
        else:
            fc8w = tf.Variable(tf.random_normal([4096, output_dim],
                                                dtype=tf.float32,
                                                stddev=1e-2), name='weights')
            fc8b = tf.Variable(tf.constant(0.0, shape=[output_dim],
                                           dtype=tf.float32), name='biases')
        fc8l = tf.nn.bias_add(tf.matmul(fc7, fc8w), fc8b)

        if with_tanh:
            fc8_t = tf.nn.tanh(fc8l)
        else:
            fc8_t = fc8l

        def val_fn1():
            concated = tf.concat([tf.expand_dims(i, 0)
                                  for i in tf.split(fc8_t, 10, 0)], 0)
            return tf.reduce_mean(concated, 0)
        fc8 = tf.cond(stage > 0, val_fn1, lambda: fc8_t)

        deep_param_img['fc8'] = [fc8w, fc8b]
        train_layers += [fc8w, fc8b]

    print("img model loading finished")
    # Return outputs
    return fc8, deep_param_img, train_layers
