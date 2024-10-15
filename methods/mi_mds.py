from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, cdist
from tqdm import tqdm
import numpy as np
from mdscuda import mds_fit, minkowski_pairs

class MI_MDS:
    def __init__(self, n_components=2, n_neighbors=5, tol=1e-4, max_iter=1000, verbose=False, cuda=False):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.verbose = verbose
        self.tol = tol
        self.max_iter = max_iter
        self.cuda = cuda

    def fit(self, X):
        self.X_train = X
        if self.cuda:
            # https://github.com/SethEBaldwin/mdscuda
            self.P_train = mds_fit(delta=minkowski_pairs(X, sqform=False), n_dims=self.n_components)
        else:
            self.P_train = MDS(n_components=self.n_components, normalized_stress='auto', n_jobs=-1).fit_transform(X)

        return self

    def transform(self):
        '''
        return the embedding of the training data
        '''
        return self.P_train

    def transform_oos(self, X_test):
        '''
        return the embedding of the test data (interpolation)
        '''
        X_train = self.X_train
        P_train = self.P_train
        k = self.n_neighbors

        if self.verbose:
            print('mi-mds interpolating: computing knn and pairwise distances')

        knn = NearestNeighbors(n_neighbors=k).fit(X_train)

        _, indi = knn.kneighbors(X_test)

        all_X = np.concatenate((X_train, X_test))

        P_knn = np.array([P_train[idxs] for idxs in indi])
        P_mean = np.mean(P_knn, axis=1)

        D_high_all = pdist(all_X, 'euclidean')
        D_high_k = [cdist(X_train[indi[:, i]], X_test, 'euclidean').diagonal()[:,np.newaxis] for i in range(k)]

        def loss_func(x_t_1):

            s = 0

            for i in range(k):
                D_high_i = D_high_k[i]
                D_low_i = cdist(P_knn[:, i], x_t_1, 'euclidean').diagonal()[:, np.newaxis]
                f = (x_t_1 - P_knn[:, i]) * (D_high_i / D_low_i)
                s += f

            x_t = P_mean + s / k

            S_t_1 = np.concatenate((P_train, x_t_1))
            S_t = np.concatenate((P_train, x_t))

            sigma_t_1 = np.sum((pdist(S_t_1, 'euclidean') - D_high_all)**2)
            sigma_t = np.sum((pdist(S_t, 'euclidean') - D_high_all)**2)

            return abs(sigma_t_1 - sigma_t), x_t

        x_t = P_mean

        _iter = tqdm(range(self.max_iter), disable=not self.verbose)
        for i in _iter:
            res, x_t = loss_func(x_t_1=x_t)
            _iter.set_postfix(loss = f"{res:.6g}")
            if res < self.tol:
                break

        return x_t
