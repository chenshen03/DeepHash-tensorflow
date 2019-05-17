import tensorflow as tf
from distance.tfversion import distance


def cross_entropy(u, label_u, alpha=0.5, normed=False):
    ''' 
    DHN
    Param: 
    Return: 
    '''
    label_ip = tf.cast(
        tf.matmul(label_u, tf.transpose(label_u)), tf.float32)
    s = tf.clip_by_value(label_ip, 0.0, 1.0)

    # compute balance param
    # s_t \in {-1, 1}
    s_t = tf.multiply(tf.add(s, tf.constant(-0.5)), tf.constant(2.0))
    sum_1 = tf.reduce_sum(s)
    sum_all = tf.reduce_sum(tf.abs(s_t))

    balance_param = tf.add(tf.abs(tf.add(s, tf.constant(-1.0))),
                            tf.multiply(tf.div(sum_all, sum_1), s))

    if normed:
        ip_1 = tf.matmul(u, tf.transpose(u))

        def reduce_shaper(t):
            return tf.reshape(tf.reduce_sum(t, 1), [tf.shape(t)[0], 1])
        mod_1 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(u)),
                                    reduce_shaper(tf.square(u)), transpose_b=True))
        ip = tf.div(ip_1, mod_1)
    else:
        ip = tf.clip_by_value(tf.matmul(u, tf.transpose(u)), -1.5e1, 1.5e1)
    ones = tf.ones([tf.shape(u)[0], tf.shape(u)[0]])
    return tf.reduce_mean(tf.multiply(tf.log(ones + tf.exp(alpha * ip)) - s * alpha * ip, balance_param))


def cauchy_cross_entropy(u, label_u, output_dim=300, v=None, label_v=None, gamma=1, normed=True):
    ''' 
    DCH
    Param: 
    Return: 
    '''
    if v is None:
        v = u
        label_v = label_u

    label_ip = tf.cast(
        tf.matmul(label_u, tf.transpose(label_v)), tf.float32)
    s = tf.clip_by_value(label_ip, 0.0, 1.0)

    if normed:
        ip_1 = tf.matmul(u, tf.transpose(v))

        def reduce_shaper(t):
            return tf.reshape(tf.reduce_sum(t, 1), [tf.shape(t)[0], 1])
        mod_1 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(u)), reduce_shaper(
            tf.square(v)) + tf.constant(0.000001), transpose_b=True))
        dist = tf.constant(np.float32(output_dim)) / 2.0 * \
            (1.0 - tf.div(ip_1, mod_1) + tf.constant(0.000001))
    else:
        r_u = tf.reshape(tf.reduce_sum(u * u, 1), [-1, 1])
        r_v = tf.reshape(tf.reduce_sum(v * v, 1), [-1, 1])

        dist = r_u - 2 * tf.matmul(u, tf.transpose(v)) + \
            tf.transpose(r_v) + tf.constant(0.001)

    cauchy = gamma / (dist + gamma)

    s_t = tf.multiply(tf.add(s, tf.constant(-0.5)), tf.constant(2.0))
    sum_1 = tf.reduce_sum(s)
    sum_all = tf.reduce_sum(tf.abs(s_t))
    balance_param = tf.add(
        tf.abs(tf.add(s, tf.constant(-1.0))), tf.multiply(tf.div(sum_all, sum_1), s))

    mask = tf.equal(tf.eye(tf.shape(u)[0]), tf.constant(0.0))

    cauchy_mask = tf.boolean_mask(cauchy, mask)
    s_mask = tf.boolean_mask(s, mask)
    balance_p_mask = tf.boolean_mask(balance_param, mask)

    all_loss = - s_mask * \
        tf.log(cauchy_mask) - (tf.constant(1.0) - s_mask) * \
        tf.log(tf.constant(1.0) - cauchy_mask)

    return tf.reduce_mean(tf.multiply(all_loss, balance_p_mask))


def triplet_loss(anchor, pos, neg, margin, dist_type='euclidean2'):
    ''' 
    DTQ
    Param: 
    Return: 
    '''
    with tf.variable_scope('triplet_loss'):
        pos_dist = distance(anchor, pos, pair=False, dist_type=dist_type)
        neg_dist = distance(anchor, neg, pair=False, dist_type=dist_type)
        basic_loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)
        loss = tf.reduce_mean(basic_loss, 0)

        tf.summary.histogram('pos_dist', pos_dist)
        tf.summary.histogram('neg_dist', neg_dist)
        tf.summary.histogram('pos_dist - neg_dist', pos_dist - neg_dist)

    return loss


def simple_quantization_loss(z):
    ''' 
    
    Param: 
    Return: 
    '''
    with tf.name_scope('simple_quantization_loss'):
        q_loss = tf.reduce_mean(tf.square(tf.subtract(tf.abs(z), tf.constant(1.0))))
    return q_loss


def quantization_loss(z, h, C):
    ''' 
    DTQ
    Param: 
    Return: 
    '''
    with tf.name_scope('quantization_loss'):
        q_loss = tf.reduce_mean(tf.reduce_sum(z - tf.matmul(h, C), -1))
    return q_loss


def square_quantization_loss(z, h, C):
    ''' 
    DQN
    Param: 
    Return: 
    '''
    with tf.name_scope('square_quantization_loss'):
        q_loss = tf.reduce_mean(tf.reduce_sum(tf.square(z - tf.matmul(h, C)), 1))
    return q_loss

