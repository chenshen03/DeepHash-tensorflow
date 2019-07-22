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


def cauchy_cross_entropy_loss(u, label_u, gamma=16, normed=True):
    '''cauchy cross entropy loss
    - Deep Cauchy Hashing for Hamming Space Retrieval
    '''
    with tf.name_scope('cauchy_cross_entropy_loss'):
        bit = tf.cast(tf.shape(u)[1], tf.float32)

        if normed:
            ip_1 = tf.matmul(u, tf.transpose(u))
            mod_1 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(u)), reduce_shaper(
                tf.square(u)) + tf.constant(1e-6), transpose_b=True))
            dist = bit / 2.0 * (1.0 - tf.div(ip_1, mod_1) + tf.constant(1e-6))
        else:
            r_u = tf.reshape(tf.reduce_sum(u * u, 1), [-1, 1])
            r_v = tf.reshape(tf.reduce_sum(u * u, 1), [-1, 1])

            dist = r_u - 2 * tf.matmul(u, tf.transpose(u)) + \
                tf.transpose(r_v) + tf.constant(0.001)

        S = tf.clip_by_value(tf.matmul(label_u, tf.transpose(label_u)), 0.0, 1.0)
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
            # TODO DELETTE! In this setting, results will be worse.
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


def exp_loss(u, label_u, alpha, wordvec=None, balanced=True):
    '''exponential loss
    '''
    with tf.name_scope('exp_loss'):
        batch_size = tf.shape(u)[0]
        bit = tf.shape(u)[1]
        mask = tf.equal(tf.eye(batch_size), tf.constant(0.0))
        S = tf.clip_by_value(tf.matmul(label_u, tf.transpose(label_u)), 0.0, 1.0)
        S_m = tf.boolean_mask(S, mask)

        # word vector
        if wordvec != None:
            wordvec_u = tf.matmul(label_u, wordvec) / tf.reduce_sum(label_u, axis=1, keepdims=True)
            W = distance(wordvec_u, dist_type='cosine')
        
        ## margin hinge-like loss
        # balanced = False
        # D = distance(u, dist_type='euclidean2')
        # E = D
        # E_m = tf.boolean_mask(E, mask)
        # loss_1 = S_m * E_m + (1 - S_m) *  tf.maximum(alpha - E_m, 0.0)

        ## double margin hinge-like loss
        # balanced = False
        # D = distance(u, dist_type='cosine')
        # E = D
        # E_m = tf.boolean_mask(E, mask)
        # loss_1 = S_m * tf.maximum(E_m - 0.3, 0.0) + (1 - S_m) *  tf.maximum(0.45 - E_m, 0.0)

        ## cauchy cross-entropy loss
        # D = distance(u, dist_type='cosine')
        # E = tf.log(1 + alpha * D)
        # E_m = tf.boolean_mask(E, mask)
        # loss_1 = S_m * E_m + (1 - S_m) * (E_m - tf.log(tf.exp(E_m) - 1 + 1e-6))

        # sigmoid
        # D = distance(u, dist_type='cosine')
        # E = tf.log(1 + tf.exp(-alpha * (1-2*D)))
        # E_m = tf.boolean_mask(E, mask)
        # loss_1 = S_m * E_m + (1 - S_m) * (E_m - tf.log(tf.exp(E_m) - 1 + 1e-6))

        ## hyper sigmoid
        balanced = False
        alpha = 9 
        belta = 20
        gamma = 1.5
        margin = 0.25
        D = distance(u, dist_type='cosine')
        E1 = tf.log(1 + tf.exp(-alpha * (1-gamma*2*D)))
        E1_m = tf.boolean_mask(E1, mask)
        loss_s1 = S_m * E1_m
        E2 = tf.log(1 + tf.exp(-alpha * (1-gamma*2*(D-margin))))
        E2_m = tf.boolean_mask(E2, mask)
        loss_s0 = (1 - S_m) * (E2_m - tf.log(tf.exp(E2_m) - 1 + 1e-6))
        loss_1 = belta * loss_s1 + loss_s0

        ## margin exp loss
        # balanced = False
        # D = distance(u, dist_type='cosine')
        # E1 = tf.exp(2* D) - 1
        # E2 = tf.exp(2 * (1 - D)) - 1
        # E1_m = tf.boolean_mask(E1, mask)
        # E2_m = tf.boolean_mask(E2, mask)
        # loss_1 = S_m * E1_m + (1 - S_m) * E2_m

        ## post-tune
        # balanced = False
        # D = distance(u, dist_type='cosine')
        # E = D
        # E_m = tf.boolean_mask(E, mask)
        # margin = 0.05
        # loss_1 = S_m * tf.maximum(E_m - alpha + margin, 0.0) + (1 - S_m) *  tf.maximum(alpha + margin - E_m, 0.0)
        # loss_1 = S_m * tf.maximum(E_m - alpha + margin, 0.0)
        # loss_1 = (1 - S_m) *  tf.maximum(alpha + margin - E_m, 0.0)

        if balanced:
            S_all = tf.cast(batch_size * (batch_size - 1), tf.float32)
            S_1 = tf.reduce_sum(S)
            balance_param = (S_all / S_1) * S + (1 - S)
            B_m= tf.boolean_mask(balance_param, mask)
            loss_1 = B_m * loss_1

        loss = tf.reduce_mean(loss_1)
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


def cos_margin_multi_label_loss(u, label_u, wordvec, bit=300, soft=True, margin=0.7):
    '''cosine margin multi label loss
    - Deep Visual-Semantic Quantization for Efficient Image Retrieval
    '''
    # N: batchsize, L: label_dim, D: 300
    # u: N * D
    # label_u: N * L
    # wordvec: L * D
    with tf.name_scope('cos_margin_multi_label_loss'):
        assert bit == 300

        batch_size = tf.cast(tf.shape(label_u)[0], tf.int32)
        n_class = tf.cast(tf.shape(label_u)[1], tf.int32)
        if soft == True:
            ip_2 = tf.matmul(u, wordvec, transpose_b=True)
            # multiply ids to inner product
            mod_2 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(
                u)), reduce_shaper(tf.square(wordvec)), transpose_b=True))
            # cos_2: N * L
            cos_2 = tf.div(ip_2, mod_2)

            # ip_3: L * L
            # compute soft margin
            ip_3 = tf.matmul(wordvec, wordvec, transpose_b=True)
            # use word_dic to avoid 0 in /
            mod_3 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(
                wordvec)), reduce_shaper(tf.square(wordvec)), transpose_b=True))
            margin_param = tf.subtract(tf.constant(
                1.0, dtype=tf.float32), tf.div(ip_3, mod_3))

            # cos - cos: N * L * L
            cos_cos_1 = tf.subtract(tf.expand_dims(margin_param, 0), tf.subtract(
                tf.expand_dims(cos_2, 2), tf.expand_dims(cos_2, 1)))
            # we need to let the wrong place be 0
            cos_cos = tf.multiply(cos_cos_1, tf.expand_dims(label_u, 2))

            cos_loss = tf.reduce_sum(tf.maximum(
                tf.constant(0, dtype=tf.float32), cos_cos))
            loss = tf.div(cos_loss, tf.multiply(tf.cast(
                n_class, dtype=tf.float32), tf.reduce_sum(label_u)))  
        else:
            margin_param = tf.constant(margin, dtype=tf.float32)

            # v_label: N * L * D
            v_label = tf.multiply(tf.expand_dims(label_u, 2), tf.expand_dims(wordvec, 0))
            # ip_1: N * L
            ip_1 = tf.reduce_sum(tf.multiply(tf.expand_dims(u, 1), v_label), 2)
            # mod_1: N * L
            v_label_mod = tf.multiply(tf.expand_dims(
                tf.ones([batch_size, n_class]), 2), tf.expand_dims(wordvec, 0))
            mod_1 = tf.sqrt(tf.multiply(tf.expand_dims(tf.reduce_sum(
                tf.square(u), 1), 1), tf.reduce_sum(tf.square(v_label_mod), 2)))
            # cos_1: N * L
            cos_1 = tf.div(ip_1, mod_1)

            ip_2 = tf.matmul(u, wordvec, transpose_b=True)
            # multiply ids to inner product
            mod_2 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(
                u)), reduce_shaper(tf.square(wordvec)), transpose_b=True))
            # cos_2: N * L
            cos_2 = tf.div(ip_2, mod_2)

            # cos - cos: N * L * L
            cos_cos_1 = tf.subtract(margin_param, tf.subtract(
                tf.expand_dims(cos_1, 2), tf.expand_dims(cos_2, 1)))
            # we need to let the wrong place be 0
            cos_cos = tf.multiply(cos_cos_1, tf.expand_dims(label_u, 2))

            cos_loss = tf.reduce_sum(tf.maximum(
                tf.constant(0, dtype=tf.float32), cos_cos))
            loss = tf.div(cos_loss, tf.multiply(tf.cast(
                n_class, dtype=tf.float32), tf.reduce_sum(label_u)))       
    return loss


'''quantization loss
'''

def quantization_loss(u, q_type='L2'):
    '''quantization loss
    - Deep Hashing Network for Efficient Similarity Retrieval
    - Deep Supervised Hashing for Fast Image Retrieval
    - Deep Cauchy Hashing for Hamming Space Retrieval
    - Deep Visual-Semantic Hashing for Cross-Modal Retrieval
    - Correlation Hashing Network for Efficient Cross-Modal Retrieval
    '''
    with tf.name_scope('quantization_loss'):
        if q_type == 'L2':
            loss = tf.reduce_mean(tf.square(tf.abs(u) - tf.constant(1.0)))
        elif q_type == 'L1':
            loss = tf.reduce_mean(tf.abs(tf.abs(u) - tf.constant(1.0)))
        elif q_type == 'cauchy':
            epsilon = 0.58
            loss = tf.reduce_mean(tf.log(1 + tf.abs((tf.abs(u) - tf.constant(1.0))) / epsilon))
        elif q_type == 'margin':
            margin = 0.5
            loss = tf.reduce_mean(tf.maximum(margin - tf.abs(u), 0.0))
        elif q_type == 'max_margin':
            bit = tf.shape(u)[1]
            margin = 0.95
            D = distance(tf.abs(u), tf.ones(bit), dist_type='cos')
            loss = tf.reduce_mean(tf.maximum(margin - D, 0.0))
    return loss


def pq_loss(u, h, C, wordvec=None, squared=True):
    '''product quantization loss
    - Deep Quantization Network for Efficient Image Retrieval
    - Deep Visual-Semantic Quantization for Efficient Image Retrieval
    - Deep Triplet Quantization
    '''
    with tf.name_scope('pq_loss'):
        dist = u - tf.matmul(h, C)

        if wordvec != None:
            dist = tf.matmul(dist, wordvec, transpose_b=True)

        if squared:
            dist = tf.square(dist)

        loss = tf.reduce_mean(tf.reduce_sum(dist, 1))
    return loss


'''balance and independence loss
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
        H_mean = tf.reduce_mean(H, axis=0)
        loss = tf.reduce_mean(tf.square(H_mean))
    return loss


def independence_loss(u):
    '''independence loss
    - Deep Triplet Quantization
    '''
    with tf.name_scope('independence_loss'):
        batch_size = tf.shape(u)[0]
        bit = tf.shape(u)[1]
        H = tf.sign(u)
        I = tf.eye(bit)
        loss = tf.reduce_mean(tf.square(tf.matmul(
            H, H, transpose_a=True) / tf.cast(batch_size, tf.float32) - I))
    return loss


'''listwise loss
- Hashing as Tie-Aware Learning to Rank
'''


'''classification loss
- Deep Semantic Hashing with Generative Adversarial Networks
- Deep Supervised Discrete Hashing
- Supervised Learning of Semantics-preserving Hashing via Deep Neural Networks for Large-scale Image Search
- Deep Supervised Cross-modal Retrieval
'''
