import tensorflow as tf
import numpy as np
from distance.tfversion import distance
from util import sign, reduce_shaper


'''pairwise loss
'''

def inner_product_loss(u, label_u, balanced=True):
    '''pairwise inner product loss
    - Hash with graph
    - Supervised Hashing for Image Retrieval via Image Representation Learning
    - Deep Discrete Supervised Hashing
    '''
    with tf.name_scope('inner_product_loss'):
        B = tf.cast(tf.shape(u)[1], tf.float32)
        ip = tf.matmul(u, u, transpose_b=True)

        # let sim = {0, 1} to be {-1, 1}
        S = tf.clip_by_value(tf.matmul(label_u, tf.transpose(label_u)), 0.0, 1.0)
        Sim = tf.multiply(tf.add(S, tf.constant(-0.5)), tf.constant(2.0))

        loss_1 = tf.square(tf.subtract(Sim, tf.div(ip, B)))

        if balanced:
            with tf.name_scope('balance'):
                sum_1 = tf.reduce_sum(S)
                sum_all = tf.reduce_sum(tf.abs(Sim))
                balance_param = tf.add(tf.abs(tf.add(S, tf.constant(-1.0))),
                                        tf.multiply(tf.div(sum_all, sum_1), S))
                loss_1 = tf.multiply(loss_1, balance_param)
        
        loss = tf.reduce_mean(loss_1)
    return loss

def cosine_loss(u, label_u, balanced=True):
    '''squared pairwise cosine loss
    - Deep Quantization Network for Efficient Image Retrieval
    '''
    with tf.name_scope('cosine_loss'):
        ip_1 = tf.matmul(u, u, transpose_b=True)
        mod_1 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(u)), reduce_shaper(
            tf.square(u)), transpose_b=True))
        cos_1 = tf.div(ip_1, mod_1)

        # let Sim = {0, 1} to be {-1, 1}
        S = tf.clip_by_value(tf.matmul(label_u, tf.transpose(label_u)), 0.0, 1.0)
        Sim = tf.multiply(tf.add(S, tf.constant(-0.5)), tf.constant(2.0))

        loss_1 = tf.square(tf.subtract(Sim, cos_1))

        if balanced:
            with tf.name_scope('balance'):
                sum_1 = tf.reduce_sum(S)
                sum_all = tf.reduce_sum(tf.abs(Sim))
                balance_param = tf.add(tf.abs(tf.add(S, tf.constant(-1.0))),
                                        tf.multiply(tf.div(sum_all, sum_1), S))
                loss_1 = tf.multiply(loss_1, balance_param)

        loss = tf.reduce_mean(loss_1)
    return loss  

def cross_entropy_loss(u, label_u, alpha=0.5, normed=True, balanced=True):
    '''cross entropy loss
    - Deep Hashing Network for Efficient Similarity Retrieval
    '''
    with tf.name_scope('cross_entropy_loss'):
        if normed:
            ip_1 = tf.matmul(u, tf.transpose(u))
            mod_1 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(u)),
                                        reduce_shaper(tf.square(u)), transpose_b=True))
            ip = tf.div(ip_1, mod_1)
        else:
            ip = tf.clip_by_value(tf.matmul(u, tf.transpose(u)), -1.5e1, 1.5e1)
            
        ones = tf.ones([tf.shape(u)[0], tf.shape(u)[0]])
        S = tf.clip_by_value(tf.matmul(label_u, tf.transpose(label_u)), 0.0, 1.0)

        loss_1 = tf.log(ones + tf.exp(alpha * ip)) - S * alpha * ip

        if balanced:
            with tf.name_scope('balance'):
                # let Sim \in {-1, 1}
                Sim = tf.multiply(tf.add(S, tf.constant(-0.5)), tf.constant(2.0))
                sum_1 = tf.reduce_sum(S)
                sum_all = tf.reduce_sum(tf.abs(Sim))
                balance_param = tf.add(tf.abs(tf.add(S, tf.constant(-1.0))),
                                        tf.multiply(tf.div(sum_all, sum_1), S))
                loss_1 = tf.multiply(loss_1, balance_param)

        loss = tf.reduce_mean(loss_1)
    return loss

def cauchy_cross_entropy_loss(u, label_u, v=None, label_v=None, output_dim=300, gamma=1, normed=True):
    '''cauchy cross entropy loss
    - Deep Cauchy Hashing for Hamming Space Retrieval
    '''
    with tf.name_scope('cauchy_cross_entropy_loss'):
        if v is None:
            v = u
            label_v = label_u

        if normed:
            ip_1 = tf.matmul(u, tf.transpose(v))
            mod_1 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(u)), reduce_shaper(
                tf.square(v)) + tf.constant(0.000001), transpose_b=True))
            dist = tf.constant(np.float32(output_dim)) / 2.0 * \
                (1.0 - tf.div(ip_1, mod_1) + tf.constant(0.000001))
        else:
            r_u = tf.reshape(tf.reduce_sum(u * u, 1), [-1, 1])
            r_v = tf.reshape(tf.reduce_sum(v * v, 1), [-1, 1])

            dist = r_u - 2 * tf.matmul(u, tf.transpose(v)) + \
                tf.transpose(r_v) + tf.constant(0.001)

        S = tf.clip_by_value(tf.matmul(label_u, tf.transpose(label_v)), 0.0, 1.0)
        with tf.name_scope('balance'):
            Sim = tf.multiply(tf.add(S, tf.constant(-0.5)), tf.constant(2.0))
            sum_1 = tf.reduce_sum(S)
            sum_all = tf.reduce_sum(tf.abs(Sim))
            balance_param = tf.add(tf.abs(tf.add(S, tf.constant(-1.0))), 
                                    tf.multiply(tf.div(sum_all, sum_1), S))

        mask = tf.equal(tf.eye(tf.shape(u)[0]), tf.constant(0.0))
        cauchy = gamma / (dist + gamma)
        cauchy_mask = tf.boolean_mask(cauchy, mask)
        s_mask = tf.boolean_mask(S, mask)
        balance_p_mask = tf.boolean_mask(balance_param, mask)

        all_loss = - s_mask * \
            tf.log(cauchy_mask) - (tf.constant(1.0) - s_mask) * \
            tf.log(tf.constant(1.0) - cauchy_mask)

        loss = tf.reduce_mean(tf.multiply(all_loss, balance_p_mask))
    return loss

def contrastive_loss(u, label_u, margin=4, balanced=False):
    '''contrastive loss
    - Deep Supervised Hashing for Fast Image Retrieval
    '''
    with tf.name_scope('contrastive_loss'):
        batch_size = tf.cast(tf.shape(u)[0], tf.float32)
        S = tf.clip_by_value(tf.matmul(label_u, tf.transpose(label_u)), 0.0, 1.0)
        dist = distance(u)

        loss_1 = S * dist + (1 - S) * tf.maximum(margin - dist, 0.0)

        if balanced:
            with tf.name_scope('balance'):
                # let Sim \in {-1, 1}
                Sim = tf.multiply(tf.add(S, tf.constant(-0.5)), tf.constant(2.0))
                sum_1 = tf.reduce_sum(S)
                sum_all = tf.reduce_sum(tf.abs(Sim))
                balance_param = tf.add(tf.abs(tf.add(S, tf.constant(-1.0))),
                                        tf.multiply(tf.div(sum_all, sum_1), S))
                loss_1 = tf.multiply(loss_1, balance_param)

        loss = tf.reduce_sum(loss_1) / (batch_size*(batch_size-1))
    return loss


'''triplet loss
'''

def triplet_loss(anchor, pos, neg, margin, dist_type='euclidean2'):
    '''triplet loss
    - Deep Triplet Quantization
    '''
    with tf.name_scope('triplet_loss'):
        pos_dist = distance(anchor, pos, pair=False, dist_type=dist_type)
        neg_dist = distance(anchor, neg, pair=False, dist_type=dist_type)
        basic_loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)
        loss = tf.reduce_mean(basic_loss, 0)

        tf.summary.histogram('pos_dist', pos_dist)
        tf.summary.histogram('neg_dist', neg_dist)
        tf.summary.histogram('pos_dist - neg_dist', pos_dist - neg_dist)
    return loss


'''listwise loss
- Hashing as Tie-Aware Learning to Rank
'''


'''quantization loss
'''

def quantization_loss(z, L2=True):
    '''quantization loss
    - Deep Hashing Network for Efficient Similarity Retrieval
    - Deep Supervised Hashing for Fast Image Retrieval
    '''
    with tf.name_scope('quantization_loss'):
        if L2:
            # L2^2 norm, DHN
            loss = tf.reduce_mean(tf.square(tf.subtract(tf.abs(z), tf.constant(1.0))))
        else:
            # L1 norm, DSH
            loss = tf.reduce_mean(tf.abs(tf.subtract(tf.abs(z), tf.constant(1.0))))
    return loss

def cauchy_quantization_loss(z):
    '''quantization loss
    - Deep Cauchy Hashing for Hamming Space Retrieval
    '''
    with tf.name_scope('cauchy_quantization_loss'):
        pass
    return loss    

def pq_loss(z, h, C, squared=False):
    '''product quantization loss
    - Deep Quantization Network for Efficient Image Retrieval
    - Deep Triplet Quantization
    '''
    with tf.name_scope('pq_loss'):
        if squared:
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(z - tf.matmul(h, C)), 1))
        else:
            loss = tf.reduce_mean(tf.reduce_sum(z - tf.matmul(h, C), 1))
    return loss


'''classification loss
- Deep Semantic Hashing with Generative Adversarial Networks
- Deep Supervised Discrete Hashing
- Supervised Learning of Semantics-preserving Hashing via Deep Neural Networks for Large-scale Image Search
'''


'''independence and balance loss
- Deep semantic ranking based hashing for multi-label image retrieval
- Supervised Learning of Semantics-preserving Hashing via Deep Neural Networks for Large-scale Image Search
'''

def balance_loss(u):
    '''balance loss

    Each bit should be half 0 and half 1.
    - Supervised Learning of Semantics-preserving Hashing via Deep Neural Networks for Large-scale Image Search
    '''
    with tf.name_scope('balance_loss'):
        H = tf.sign(u)
        H_mean = tf.reduce_mean(H, axis=1)
        loss = tf.reduce_mean(tf.square(H_mean))
    return loss
