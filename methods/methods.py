import yaml

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, MDS
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch

from umap.parametric_umap import ParametricUMAP
from umap import UMAP

from .ptsne.ptsne import ParametricTSNE
from .parametric_dr import TSNE_NN
from .cdr import CDRModelHandler

from .autoencoder import AutoEncoder
from ivis import Ivis
from .dlmp import train_model

from .oos_mds import oos_MDS
from .oos_isomap import oos_Isomap
from .mi_mds import MI_MDS

from .landmark import LandmarkMDS, LandmarkIsomap
from .kernel_base import KernelMDS, KernalIsomap, KernelTSNE

from .ptsne09 import PTSNE09

import numpy as np

class Parameters:
    def __init__(self):
        with open('parameters.yml', 'r') as f:
            self.__params__ = yaml.load(f, Loader=yaml.FullLoader)

    def available(self):
        return list(self.__params__.keys())

    def available_methods(self, parameter_name:str):
        if parameter_name not in self.__params__:
            raise ValueError(f"Unknown parameter '{parameter_name}'")
        return list(self.__params__[parameter_name].keys())

    def get_all(self):
        return self.__params__

    def get(self, parameter_name:str, method_name:str):
        if parameter_name not in self.__params__:
            raise ValueError(f"Unknown parameter '{parameter_name}'")
        methods_of_param = self.__params__[parameter_name]
        if method_name not in methods_of_param:
            raise ValueError(f"Unknown method '{method_name}' with parameter '{parameter_name}")
        param = methods_of_param[method_name]
        return param

class _pca:
    def fit(self, X:np.ndarray):
        self.m = PCA(n_components=2)
        self.m.fit(X)

    def transform(self, X:np.ndarray):
        if self.m is None:
            raise ValueError("Model not initialized")
        return self.m.transform(X)

    def transform_oos(self, X:np.ndarray):
        return self.transform(X)

class _ptsne:

    def __init__(self, params, verbose=False):
        self.perplexities = params.get('perplexities', 'ptsne')
        self.epochs = params.get('epochs', 'ptsne')
        self.lr = params.get('lr', 'ptsne')
        self.verbose = verbose

    def fit(self, X:np.ndarray):
        self.std = StandardScaler()
        X_train = self.std.fit_transform(X)

        # LEGACY
        self.m = ParametricTSNE(input_dim=X_train.shape[1], output_dim=2, perp=self.perplexities, seed=42, use_cuda=True)
        self.m.fit(training_data=torch.tensor(X_train, dtype=torch.float).cuda(), learning_rate=self.lr, epochs=self.epochs, verbose=self.verbose, batch_size=int(X.shape[0] / 10 if X.shape[0] > 256 else X.shape[0]))

    def transform(self, X:np.ndarray):
        if self.m is None:
            raise ValueError("Model not initialized")
        X_test = self.std.transform(X)

        # LEGACY
        return self.m(torch.tensor(X_test, dtype=torch.float).cuda()).cpu().detach().numpy()
    
    def transform_oos(self, X:np.ndarray):
        return self.transform(X)

class _ptsne_new:
    
    def __init__(self, params, verbose=False):
        self.perplexities = params.get('perplexities', 'ptsne_new')
        self.epochs = params.get('epochs', 'ptsne_new')
        self.lr = params.get('lr', 'ptsne_new')
        self.verbose = verbose

    def fit(self, X:np.ndarray):
        self.std = StandardScaler()
        X_train = self.std.fit_transform(X)

        # NEW!!
        self.m = PTSNE09(lr=self.lr, perplexity=self.perplexities, n_epochs=self.epochs, verbose=self.verbose, device='cuda', batch_size=int(X_train.shape[0] / 10 if X_train.shape[0] > 200 else 200))
        self.m.fit(X_train)

    def transform(self, X:np.ndarray):
        if self.m is None:
            raise ValueError("Model not initialized")
        X_test = self.std.transform(X)

        # NEW!!
        return self.m.transform(X_test)

    def transform_oos(self, X:np.ndarray):
        return self.transform(X)


class _pumap:

    def __init__(self, params, verbose=False):
        self.knn = params.get('knn', 'pumap')
        self.verbose = verbose

    def fit(self, X:np.ndarray):
        self.m = ParametricUMAP(n_neighbors=self.knn, loss_report_frequency=1, verbose=1 if self.verbose else 0, batch_size=int(X.shape[0] / 10 if X.shape[0] > 256 else X.shape[0]))
        self.m.fit(X)

    def transform(self, X:np.ndarray):
        if self.m is None:
            raise ValueError("Model not initialized")
        return self.m.transform(X)

    def transform_oos(self, X:np.ndarray):
        return self.transform(X)


class _autoencoder:
    def __init__(self, params, verbose=False):
        self.epochs = params.get('epochs', 'autoencoder')
        self.lr = params.get('lr', 'autoencoder')
        self.verbose = verbose

    def fit(self, X:np.ndarray):
        self.m = AutoEncoder(input_dim=X.shape[1], device="cuda")
        self.std = StandardScaler()
        XX = self.std.fit_transform(X)
        self.m.fit(data=XX, n_epochs=self.epochs, verbose=self.verbose, lr=self.lr)

    def transform(self, X:np.ndarray):
        if self.m is None:
            raise ValueError("Model not initialized")
        return self.m.transform(self.std.transform(X))

    def transform_oos(self, X:np.ndarray):
        return self.transform(X)

class _dlmp_tsne:
    """
        [Deep learning multidimensional projections](https://github.com/mespadoto/dlmp)
    """
    def __init__(self, params, verbose=False):
        self.epochs = params.get('epochs', 'dlmp-tsne')
        self.perplexities = params.get('perplexities', 'dlmp-tsne')
        self.lr = params.get('lr', 'dlmp-tsne')
        self.verbose = verbose

    def fit(self, X:np.ndarray):
        tsne = TSNE(perplexity=self.perplexities)
        self.std = StandardScaler()
        XX = self.std.fit_transform(X)
        self.X_2d = tsne.fit_transform(XX)
        self.X_2d = MinMaxScaler().fit_transform(self.X_2d)
        self.m, _ = train_model(XX, self.X_2d, epochs=self.epochs, verbose=self.verbose, lr=self.lr, batch_size=int(X.shape[0] / 10 if X.shape[0] > 256 else X.shape[0]))

    def transform(self, X:np.ndarray):
        if self.m is None:
            raise ValueError("Model not initialized")
        return self.X_2d

    def transform_oos(self, X:np.ndarray):
        if self.m is None:
            raise ValueError("Model not initialized")
        return self.m.predict(self.std.transform(X), verbose=self.verbose)

class _dlmp_umap:
    """
        [Deep learning multidimensional projections](https://github.com/mespadoto/dlmp)
    """
    def __init__(self, params, verbose=False):
        self.epochs = params.get('epochs', 'dlmp-umap')
        self.knn = params.get('knn', 'dlmp-umap')
        self.lr = params.get('lr', 'dlmp-umap')
        self.verbose = verbose

    def fit(self, X:np.ndarray):
        umap = UMAP(n_neighbors=self.knn, min_dist=0.1)
        self.std = StandardScaler()
        XX = self.std.fit_transform(X)
        self.X_2d = umap.fit_transform(XX)
        self.X_2d = MinMaxScaler().fit_transform(self.X_2d)
        self.m, _ = train_model(XX, self.X_2d, epochs=self.epochs, verbose=self.verbose, lr=self.lr, batch_size=int(X.shape[0] / 10 if X.shape[0] > 256 else X.shape[0]))

    def transform(self, X:np.ndarray):
        if self.m is None:
            raise ValueError("Model not initialized")
        return self.X_2d

    def transform_oos(self, X:np.ndarray):
        if self.m is None:
            raise ValueError("Model not initialized")
        return self.m.predict(self.std.transform(X), verbose=self.verbose)

class _ptsne22:
    def __init__(self, params:Parameters, verbose=False):
        self.perplexities = params.get('perplexities', 'ptsne22')
        self.epochs = params.get('epochs', 'ptsne22')
        self.device = torch.device('cuda')
        self.verbose = verbose

    def fit(self, X:np.ndarray):
        self.m = TSNE_NN(self.device, n_epochs=self.epochs, verbose = self.verbose, batch_size=int(X.shape[0] / 10 if X.shape[0] > 256 else X.shape[0]))
        self.m.perplexity = self.perplexities
        self.m.fit(X)

    def transform(self, X:np.ndarray):
        if self.m is None:
            raise ValueError("Model not initialized")
        return self.m.fit_val(X)

    def transform_oos(self, X:np.ndarray):
        return self.transform(X)

class _cdr:
    def __init__(self, params: Parameters, verbose=False) -> None:
        self.epochs = params.get('epochs', 'cdr')
        self.knn = params.get('knn', 'cdr')
        self.lr = params.get('lr', 'cdr')
        self.verbose = verbose

    def fit(self, X:np.ndarray):
        self.std = StandardScaler().fit(X)
        XX = self.std.transform(X)
        self.m = CDRModelHandler(device='cuda', n_neighbors=self.knn, epoch_nums=self.epochs, LR=self.lr)
        self.m.fit(XX, verbose=self.verbose, batch_size=int(X.shape[0] / 10 if X.shape[0] > 256 else X.shape[0]))

    def transform(self, X:np.ndarray):
        if self.m is None:
            raise ValueError("Model not initialized")
        XX = self.std.transform(X)
        return self.m.transform(XX)

    def transform_oos(self, X:np.ndarray):
        return self.transform(X)

class _ivis:
    def __init__(self, params: Parameters, verbose=False) -> None:
        self.verbose = verbose

    def fit(self, X:np.ndarray):
        self.m = Ivis(embedding_dims=2, epochs=1000, verbose=self.verbose, batch_size=int(X.shape[0] / 10 if X.shape[0] > 256 else X.shape[0]))
        self.m.fit(X)

    def transform(self, X:np.ndarray):
        if self.m is None:
            raise ValueError("Model not initialized")
        return self.m.transform(X)

    def transform_oos(self, X:np.ndarray):
        return self.transform(X)

class _oos_base:

    def transform(self, X:np.ndarray):
        if self.m is None:
            raise ValueError("Model not initialized")
        return self.m.transform()

    def transform_oos(self, X:np.ndarray):
        if self.m is None:
            raise ValueError("Model not initialized")
        return self.m.transform_oos(X)


class _oos_mds(_oos_base):

    def fit(self, X:np.ndarray):
        self.m = oos_MDS(n_components=2).fit(X)

class _oos_isomap(_oos_base):
    def __init__(self, params: Parameters) -> None:
        self.knn = params.get('knn', 'oos-isomap')

    def fit(self, X:np.ndarray):
        self.m = oos_Isomap(n_components=2).fit(X)

class _landmark_mds(_oos_base):

    def fit(self, X:np.ndarray):
        self.m = LandmarkMDS().fit(X)

class _landmark_isomap(_oos_base):
    def __init__(self, params: Parameters):
        self.knn = params.get('knn', 'lisomap')

    def fit(self, X:np.ndarray):
        self.m = LandmarkIsomap(n_neighbors=self.knn).fit(X)

class _kernel_mds(_oos_base):

    def fit(self, X:np.ndarray):
        self.m = KernelMDS().fit(X)

class _kernel_isomap(_oos_base):
    def __init__(self, params: Parameters):
        self.knn = params.get('knn', 'kisomap')

    def fit(self, X:np.ndarray):
        self.m = KernalIsomap(n_neighbors=self.knn).fit(X)

class _kernel_tsne(_oos_base):
    def __init__(self, params: Parameters):
        self.perplexities = params.get('perplexities', 'ktsne')

    def fit(self, X:np.ndarray):
        self.m = KernelTSNE(perplexity=self.perplexities).fit(X)


class _mi_mds(_oos_base):
    def __init__(self, params: Parameters, verbose: bool = False) -> None:
        self.knn = params.get('knn', 'mi-mds')
        self.epochs = params.get('epochs', 'mi-mds')
        self.verbose = verbose

    def fit(self, X:np.ndarray):
        self.m = MI_MDS(n_components=2, n_neighbors=self.knn, tol=0.0001, max_iter=self.epochs, verbose=self.verbose).fit(X)


class Methods:

    def __init__(self, verbose: bool = False):
        self.__params__ = Parameters()
        self.verbose = verbose
        self.methods = {
            'pca': _pca(),

            'ae': _autoencoder(self.__params__, self.verbose),
            'cdr': _cdr(self.__params__, self.verbose),
            # 'ivis': _ivis(self.__params__),
            'dlmp-tsne': _dlmp_tsne(self.__params__, self.verbose),
            'dlmp-umap': _dlmp_umap(self.__params__, self.verbose),

            'ptsne': _ptsne(self.__params__, self.verbose),
            'ptsne22': _ptsne22(self.__params__, self.verbose),
            'pumap': _pumap(self.__params__, self.verbose),

            'oos-mds': _oos_mds(),
            'oos-isomap': _oos_isomap(self.__params__),

            'mimds': _mi_mds(self.__params__, self.verbose),

            'kmds': _kernel_mds(),
            'kisomap': _kernel_isomap(self.__params__),
            'ktsne': _kernel_tsne(self.__params__),

            'lmds': _landmark_mds(),
            'lisomap': _landmark_isomap(self.__params__),

        }
        self.__params__ = None

    def available(self):
        return list(self.methods.keys())

    def parameters(self):
        return self.__params__

    def get(self, name: str):
        if name not in self.methods:
            raise ValueError(f"Unknown method: {name}")
        return self.methods[name]


