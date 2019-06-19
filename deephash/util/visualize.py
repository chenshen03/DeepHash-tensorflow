import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from scipy.spatial import distance
from sklearn import manifold


def plot_distribution(data, path):
    N, D = data.shape
    plt.figure(figsize=(32, D));
    for i in range(1, D+1):
        plt.subplot(D//4, 4, i);
        commutes = pd.Series(data[:, i-1])
        commutes.plot.hist(grid=True, bins=200, rwidth=0.9, color='#607c8e');
        plt.title(f'{i}bit')
    plt.savefig(f"{path}/data_distribution.png")

    res = ''
    for i in range(1, 11):
        t = i / 10
        ratio = (np.sum(data.flatten()>=t) + np.sum(data.flatten()<=-t)) / (N * D)
        res += f'threshold: {t:.1f}, quantizaion ratio: {ratio:.5f}\n'
    return res
    

def plot_distance(db_feats, db_label, query_feats, query_label, path):
    S = np.matmul(db_label, query_label.transpose())
    N = np.sum(S)

    plt.figure(figsize=[16, 6])

    plt.subplot(121)
    cosine_32bit = distance.cdist(db_feats, query_feats, metric='cosine')
    plt.title('cosine distribution on 32bit')
    commutes = pd.Series(np.hstack((np.random.choice(cosine_32bit[S==1].flatten(), N), \
                                    np.random.choice(cosine_32bit[S==0].flatten(), N))))
    commutes.plot.hist(grid=True, bins=200, rwidth=0.9, color='#607c8e');

    plt.subplot(122)
    euclidean_32bit = distance.cdist(db_feats, query_feats, metric='euclidean')
    plt.title('euclidean distribution on 32bit')
    commutes = pd.Series(np.hstack((np.random.choice(euclidean_32bit[S==1].flatten(), N), \
                                    np.random.choice(euclidean_32bit[S==0].flatten(), N))))
    commutes.plot.hist(grid=True, bins=200, rwidth=0.9, color='#607c8e');
    
    plt.savefig(f"{path}/distance_distribution.png")


def plot_tsne(data, label, path, R=2000):
    if label.ndim > 1:
        label = label.argmax(axis=1)
    plt.figure(figsize=(16, 12));
    embed = TSNE(n_components=2, perplexity=30, lr=1, eps=1e-9, n_iter=2000, device='cuda').fit_transform(data[:R])
    plt.scatter(embed[:, 0], embed[:, 1], c=label[:R], s=10)
    plt.savefig(f"{path}/data_t-SNE.png")


class TSNE(object):
    
    def __init__(self, n_components=2, perplexity=30, lr=1, eps=1e-9, n_iter=2000, device='cpu'):
        self.perplexity = perplexity
        self.lr = lr
        self.eps = eps
        self.n_iter = n_iter
        self.device = device
        self.n_components = n_components
    
    def t_distribution(self, y):
        n = y.shape[0]
        dist = torch.sum((y.reshape(n, 1, -1) - y.reshape(1, n, -1)) ** 2, -1)
        affinity = 1 / (1 + dist)
        affinity *= (1 - torch.eye(n, device=self.device))  # set diag to zero
        q = affinity / affinity.sum() + self.eps
        return q
    
    def fit_transform(self, x):
        dist2 = distance.squareform(distance.pdist(x, metric='sqeuclidean'))
        p = distance.squareform(manifold.t_sne._joint_probabilities(dist2, self.perplexity, False)) + self.eps

        p = torch.tensor(p, device=self.device, dtype=torch.float32).reshape(-1)
        log_p = torch.log(p)

        y = torch.randn([dist2.shape[0], self.n_components], device=self.device, requires_grad=True)
        optimizer = optim.Adam([y], lr=self.lr)
        criterion = torch.nn.KLDivLoss()

        for i_iter in range(self.n_iter):
            q = self.t_distribution(y).reshape(-1)
            loss =  (p * (log_p - torch.log(q))).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return y.detach().cpu().numpy()