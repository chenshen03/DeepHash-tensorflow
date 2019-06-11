from sklearn import manifold, datasets
from scipy.spatial import distance

import torch
import torch.optim as optim
import torch.nn.functional as F
import time


class TSNE(object):
    """ t-SNE visualization
    
    example:
    ``` python
        digits = datasets.load_digits(n_class=10)
        embed = TSNE(n_components=2, perplexity=30, lr=1, eps=1e-9, 
                    n_iter=2000, device='cuda').fit_transform(digits.data)
        plt.scatter(embed[:, 0], embed[:, 1], c=digits.target, s=10)
        plt.show();
    ```
    """
    
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

        t = time.time()
        for i_iter in range(self.n_iter):
            q = self.t_distribution(y).reshape(-1)
            loss =  (p * (log_p - torch.log(q))).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return y.detach().cpu().numpy()
