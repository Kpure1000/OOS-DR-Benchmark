from sklearn.manifold import TSNE, Isomap, MDS
from scipy.spatial.distance import pdist, squareform
import numpy as np
from scipy.spatial.distance import cdist

class _KernelBase_:

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def transform(self):
        if not hasattr(self, 'y_train'):
            raise Exception('Not fit yet.')
        return self.y_train

    def transform_oos(self, X_test):
        if not hasattr(self, 'y_train'):
            raise Exception('Not fit yet.')
        Dtr = squareform(pdist(self.X_train))

        a = Dtr.reshape(1, -1).copy()
        a.sort()
        sigma = 1 / np.sqrt(-2 * np.log(0.6)) * np.median(a[0, -10:])

        K_train = np.exp(-1 * Dtr**2 / (2 * sigma**2))
        K_train = K_train / np.sum(K_train, axis=1, keepdims=True)

        A = np.dot(np.linalg.pinv(K_train), self.y_train)

        Dtest = squareform(pdist(np.vstack((X_test, self.X_train)))) 
        Dtest = Dtest[0:X_test.shape[0], -self.X_train.shape[0]:] # dist test to train

        K_test = np.exp(-1 * Dtest**2 / (2 * sigma**2))
        K_test = K_test / np.sum(K_test, axis=1, keepdims=True)

        return np.dot(K_test, A)


class KernelMDS(_KernelBase_):
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        y_train = MDS(n_components=self.n_components, normalized_stress='auto', n_jobs=-1).fit_transform(X)
        super().fit(X, y_train)

        return self

class KernalIsomap(_KernelBase_):
    def __init__(self, n_components=2, n_neighbors=5):
        self.n_components = n_components
        self.n_neighbors = n_neighbors

    def fit(self, X):
        y_train = Isomap(n_components=self.n_components, n_neighbors=self.n_neighbors).fit_transform(X)
        super().fit(X, y_train)
        
        return self

        
class KernelTSNE(_KernelBase_):
    def __init__(self, n_components=2, perplexity=30):
        self.n_components = n_components
        self.perplexity = perplexity

    def fit(self, X):
        y_train = TSNE(n_components=self.n_components, perplexity=self.perplexity).fit_transform(X)
        super().fit(X, y_train)

        return self