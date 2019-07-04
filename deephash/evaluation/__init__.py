import numpy as np

from distance.npversion import distance
from scipy.special import comb
from util import sign

    
def get_RAMAP(q_output, q_labels, db_output, db_labels, cost=False):
    ''' 
    - On the Evaluation Metric for Hashing
    '''
    M, Q = q_output.shape
    R = Q
    RAAPs = []
    time_costs = [comb(Q, r) for r in range(Q+1)]
    distH = distance(q_output, db_output, pair=False, dist_type='hamming')
    gnds = np.dot(q_labels, db_labels.transpose()) > 0
    for i in range(M):
        gnd = gnds[i,:]
        hamm = distH[i,:]
        RAAP = 0
        for r in range(R+1):
            hamm_r_idx = np.where(hamm<=r)
            rel = len(hamm_r_idx[0])
            if(rel == 0):
                continue
            imatch = np.sum(gnd[hamm_r_idx])
            if cost:
                time_cost = np.sum(time_costs[:r+1])
                RAAP += (imatch / (rel * time_cost))
            else:
                RAAP += (imatch / rel)
        RAAP = RAAP / (R + 1)
        RAAPs.append(RAAP)
    return np.mean(RAAPs)


def whrank(features, labels):
    N, D = features.shape
    classes = np.unique(labels)
    pairnum = N
    diffvals = np.zeros((pairnum, D))
    for i in range(pairnum):
        clsid = np.random.choice(classes, 1)
        sampids = np.where(labels == clsid)[0]
        samps = np.random.permutation(sampids)[:2]
        diffvals[i] = features[samps[0], :] - features[samps[1], :]
    fmu = np.mean(diffvals, axis=0)
    fstd = np.std(diffvals, axis=0)
    return fmu, fstd


def whrankHamm(q_codes, db_codes, q_feats, fmu, fstd, w_type='ones'):
    if w_type == 'ones':
        weights = np.ones_like(q_feats)
    elif w_type == 'q':
        weights = np.abs(q_feats)
    elif w_type == 'std':
        weights = np.ones_like(q_feats) / fstd
    elif w_type == 'q_std':
        weights = np.abs(q_feats) / fstd
    elif w_type == 'erf':
        Pr = 0.5 * (1 + q_codes * np.erf((-q_feats-fmu) / (np.sqrt(2)*fstd)))
        weights = np.log((1 - Pr) / Pr)
            
    num1 = q_codes.shape[0]
    num2 = db_codes.shape[0]
    distMat = np.zeros((num1, num2))
    for i in range(num1):
        codediff = np.abs(np.tile(q_codes[i], (num2, 1)) - db_codes) / 2
        distMat[i] = np.dot(weights[i], codediff.transpose())
    return distMat


def get_whrank_mAP(q_features, q_output, q_labels, db_features, db_output, db_labels, Rs=54000):
    fmu, fstd = whrank(db_features, np.argmax(db_labels, axis=1))
    dist = whrankHamm(q_output, db_output, q_features, fmu, fstd, w_type='erf')
    unsorted_ids = np.argpartition(dist, Rs - 1)[:, :Rs]
    APx = []
    for i in range(dist.shape[0]):
        label = q_labels[i, :]
        label[label == 0] = -1
        idx = unsorted_ids[i, :]
        idx = idx[np.argsort(dist[i, :][idx])]
        imatch = np.sum(np.equal(db_labels[idx[0: Rs], :], label), 1) > 0
        rel = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, Rs + 1, 1)
        if rel != 0:
            APx.append(np.sum(Px * imatch) / rel)
    return np.mean(np.array(APx))


def finetune_distID(dist, q_features, db_features):
    N, D = q_features.shape
    distID_finetune = np.zeros_like(dist)
    for i in range(N):
        cur = 0
        for j in range(D+1):
            idx = np.where(dist[i] == j)[0]
            num = len(idx)
            if num > 1:
                d = distance(q_features[i], db_features[idx], dist_type='inner_product', pair=True)
                idx = idx[np.argsort(d)]
            distID_finetune[i,cur:cur+num] = idx
            cur += num
    distID_finetune = distID_finetune.astype(int)
    return distID_finetune


def get_finetune_mAP(q_features, q_output, q_labels, db_features, db_output, db_labels, Rs=54000):
    dist_raw = distance(q_output, db_output, pair=False, dist_type='hamming')
    dist_finetune_idx = finetune_distID(dist_raw, q_features, db_features)

    N = dist_raw.shape[0]
    dist_idx = dist_finetune_idx
    APx = []
    for i in range(N):
        label = q_labels[i, :]
        label[label == 0] = -1
        idx = dist_idx[i, :]
        imatch = np.sum(np.equal(db_labels[idx[0: Rs], :], label), 1) > 0
        rel = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, Rs + 1, 1)
        if rel != 0:
            APx.append(np.sum(Px * imatch) / rel)
    mAP = np.mean(np.array(APx))
    return mAP


# optimized
def get_mAPs(q_output, q_labels, db_output, db_labels, Rs, dist_type='inner_product'):
    dist = distance(q_output, db_output, dist_type=dist_type, pair=True)
    unsorted_ids = np.argpartition(dist, Rs - 1)[:, :Rs]
    APx = []
    for i in range(dist.shape[0]):
        label = q_labels[i, :]
        label[label == 0] = -1
        idx = unsorted_ids[i, :]
        idx = idx[np.argsort(dist[i, :][idx])]
        imatch = np.sum(np.equal(db_labels[idx[0: Rs], :], label), 1) > 0
        rel = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, Rs + 1, 1)
        if rel != 0:
            APx.append(np.sum(Px * imatch) / rel)
    return np.mean(np.array(APx))


def get_mAPs_rerank(q_output, q_labels, db_output, db_labels, Rs, dist_type='inner_product'):
    query_output = sign(q_output)
    database_output = sign(db_output)

    bit_n = query_output.shape[1]

    ips = np.dot(query_output, database_output.T)
    ips = (bit_n - ips) / 2

    mAPX = []
    query_labels = q_labels
    database_labels = db_labels
    for i in range(ips.shape[0]):
        label = query_labels[i, :]
        label[label == 0] = -1

        imatch = np.array([])
        for j in range(bit_n):
            idx = np.reshape(np.argwhere(np.equal(ips[i, :], j)), (-1))
            all_num = len(idx)

            if all_num != 0:
                ips_trad = np.dot(q_output[i, :], db_output[idx[:], :].T)
                ids_trad = np.argsort(-ips_trad, axis=0)
                db_labels_1 = database_labels[idx[:], :]

                imatch = np.append(imatch, np.sum(
                    np.equal(db_labels_1[ids_trad, :], label), 1) > 0)
                if imatch.shape[0] > Rs:
                    break

        imatch = imatch[0:Rs]
        rel = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, Rs + 1, 1)
        if rel != 0:
            mAPX.append(np.sum(Px * imatch) / rel)

    return np.mean(np.array(mAPX))


class MAPs:
    def __init__(self, R):
        self.R = R

    def get_mAPs_by_feature(self, database, query, Rs=None, dist_type='inner_product'):
        if Rs is None:
            Rs = self.R
        return get_mAPs(query.output, query.label, database.output, database.label, Rs, dist_type)

    def get_mAPs_after_sign(self, database, query, Rs=None, dist_type='inner_product'):
        if Rs is None:
            Rs = self.R
        q_output = sign(query.output)
        db_output = sign(database.output)
        return get_mAPs(q_output, query.label, db_output, database.label, Rs, dist_type)

    def get_RAMAP_after_sign(self, database, query):
        q_output = sign(query.output)
        db_output = sign(database.output)
        return get_RAMAP(q_output, query.label, db_output, database.label)

    def get_mAPs_after_sign_with_feature_rerank(self, database, query, Rs=None, dist_type='inner_product'):
        if Rs is None:
            Rs = self.R
        return get_mAPs_rerank(query.output, query.label, database.output, database.label, Rs, dist_type)

    @staticmethod
    def get_precision_recall_by_Hamming_Radius(database, query, radius=2):
        query_output = sign(query.output)
        database_output = sign(database.output)

        bit_n = query_output.shape[1]

        ips = np.dot(query_output, database_output.T)
        ips = (bit_n - ips) / 2
        ids = np.argsort(ips, 1)

        precX = []
        recX = []
        mAPX = []
        query_labels = query.label
        database_labels = database.label

        for i in range(ips.shape[0]):
            label = query_labels[i, :]
            label[label == 0] = -1
            idx = np.reshape(np.argwhere(ips[i, :] <= radius), (-1))
            all_num = len(idx)

            if all_num != 0:
                imatch = np.sum(database_labels[idx[:], :] == label, 1) > 0
                match_num = np.sum(imatch)
                precX.append(np.float(match_num) / all_num)

                all_sim_num = np.sum(
                    np.sum(database_labels[:, :] == label, 1) > 0)
                recX.append(np.float(match_num) / all_sim_num)

                if radius < 10:
                    ips_trad = np.dot(
                        query.output[i, :], database.output[ids[i, 0:all_num], :].T)
                    ids_trad = np.argsort(-ips_trad, axis=0)
                    db_labels = database_labels[ids[i, 0:all_num], :]

                    rel = match_num
                    imatch = np.sum(db_labels[ids_trad, :] == label, 1) > 0
                    Lx = np.cumsum(imatch)
                    Px = Lx.astype(float) / np.arange(1, all_num + 1, 1)
                    if rel != 0:
                        mAPX.append(np.sum(Px * imatch) / rel)
                else:
                    mAPX.append(np.float(match_num) / all_num)

            else:
                precX.append(np.float(0.0))
                recX.append(np.float(0.0))
                mAPX.append(np.float(0.0))

        return np.mean(np.array(precX)), np.mean(np.array(recX)), np.mean(np.array(mAPX))

    @staticmethod
    def get_precision_recall_by_Hamming_Radius_All(database, query):
        query_output = sign(query.output)
        database_output = sign(database.output)

        bit_n = query_output.shape[1]

        ips = np.dot(query_output, database_output.T)
        ips = (bit_n - ips) / 2
        precX = np.zeros((ips.shape[0], bit_n + 1))
        recX = np.zeros((ips.shape[0], bit_n + 1))
        mAPX = np.zeros((ips.shape[0], bit_n + 1))

        query_labels = query.label
        database_labels = database.label

        ids = np.argsort(ips, 1)

        for i in range(ips.shape[0]):
            label = query_labels[i, :]
            label[label == 0] = -1

            idx = ids[i, :]
            imatch = np.sum(database_labels[idx[:], :] == label, 1) > 0
            all_sim_num = np.sum(imatch)

            counts = np.bincount(ips[i, :].astype(np.int64))

            for r in range(bit_n + 1):
                if r >= len(counts):
                    precX[i, r] = precX[i, r - 1]
                    recX[i, r] = recX[i, r - 1]
                    mAPX[i, r] = mAPX[i, r - 1]
                    continue

                all_num = np.sum(counts[0:r + 1])

                if all_num != 0:
                    match_num = np.sum(imatch[0:all_num])
                    precX[i, r] = np.float(match_num) / all_num
                    recX[i, r] = np.float(match_num) / all_sim_num

                    rel = match_num
                    Lx = np.cumsum(imatch[0:all_num])
                    Px = Lx.astype(float) / np.arange(1, all_num + 1, 1)
                    if rel != 0:
                        mAPX[i, r] = np.sum(Px * imatch[0:all_num]) / rel
        return np.mean(np.array(precX), 0), np.mean(np.array(recX), 0), np.mean(np.array(mAPX), 0)


class MAPs_CQ:
    def __init__(self, C, subspace_num, subcenter_num, R):
        self.C = C
        self.subspace_num = subspace_num
        self.subcenter_num = subcenter_num
        self.R = R

    def get_mAPs_SQD(self, database, query, Rs=None, dist_type='inner_product'):
        if Rs is None:
            Rs = self.R
        q_output = np.dot(query.codes, self.C)
        db_output = np.dot(database.codes, self.C)
        return get_mAPs(q_output, query.label, db_output, database.label, Rs, dist_type)

    def get_mAPs_AQD(self, database, query, Rs=None, dist_type='inner_product'):
        if Rs is None:
            Rs = self.R
        q_output = query.output
        db_output = np.dot(database.codes, self.C)
        return get_mAPs(q_output, query.label, db_output, database.label, Rs, dist_type)

    def get_mAPs_by_feature(self, database, query, Rs=None, dist_type='inner_product'):
        if Rs is None:
            Rs = self.R
        q_output = query.output
        db_output = database.output
        return get_mAPs(q_output, query.label, db_output, database.label, Rs, dist_type)

    def get_mAPs_after_sign(self, database, query, Rs=None, dist_type='inner_product'):
        if Rs is None:
            Rs = self.R
        q_output = sign(query.output)
        db_output = sign(database.output)
        return get_mAPs(q_output, query.label, db_output, database.label, Rs, dist_type)


if __name__ == "__main__":
    m = MAPs(4)
    radius = 2

    class ds:
        def __init__(self):
            self.output = []
            self.label = []
    database = ds()
    query = ds()

    database.output = np.sign(np.random.rand(10000, 64) - 0.5)
    database.label = np.sign(np.random.rand(10000, 20) - 0.5)
    database.label[database.label < 0] = 0
    query.output = np.sign(np.random.rand(1000, 64) - 0.5)
    query.label = np.sign(np.random.rand(1000, 20) - 0.5)
    query.label[query.label < 0] = 0

    print(m.get_mAPs_after_sign_with_feature_rerank(database, query, 500))
    print(m.get_mAPs_by_feature(database, query, 500))
    prec, rec, maps = m.get_precision_recall_by_Hamming_Radius_All(
        database, query)
    print(prec)
    print(rec)
    print(maps)
