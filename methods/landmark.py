# Author: Danilo Motta  -- <ddanilomotta@gmail.com>

# This is an implementation of the technique described in:
# Sparse multidimensional scaling using landmark points
# http://graphics.stanford.edu/courses/cs468-05-winter/Papers/Landmarks/Silva_landmarks5.pdf

import scipy as sp
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import  StandardScaler, MinMaxScaler
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import shortest_path, connected_components
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

def MDS(D,dim=[]):
	# Number of points
	n = len(D)

	# Centering matrix
	H = - np.ones((n, n))/n
	np.fill_diagonal(H,1-1/n)
	# YY^T
	H = -H.dot(D**2).dot(H)/2

	# Diagonalize
	evals, evecs = np.linalg.eigh(H)

	# Sort by eigenvalue in descending order
	idx   = np.argsort(evals)[::-1]
	evals = evals[idx]
	evecs = evecs[:,idx]

	# Compute the coordinates using positive-eigenvalued components only
	w, = np.where(evals > 0)
	if dim!=[]:
		arr = evals
		w = arr.argsort()[-dim:][::-1]
		if np.any(evals[w]<0):
			raise Exception('Error: Not enough positive eigenvalues for the selected dim.')
	L = np.diag(np.sqrt(evals[w]))
	V = evecs[:,w]
	Y = V.dot(L)
	return Y

def landmark_MDS(D, lands, dim):
	Dl = D[:,lands]
	n = len(Dl)

	# Centering matrix
	H = - np.ones((n, n))/n
	np.fill_diagonal(H,1-1/n)
	# YY^T
	H = -H.dot(Dl**2).dot(H)/2

	# Diagonalize
	evals, evecs = np.linalg.eigh(H)

	# Sort by eigenvalue in descending order
	idx   = np.argsort(evals)[::-1]
	evals = evals[idx]
	evecs = evecs[:,idx]

	# Compute the coordinates using positive-eigenvalued components only
	w, = np.where(evals > 0)
	if dim:
		arr = evals
		w = arr.argsort()[-dim:][::-1]
		if np.any(evals[w]<0):
			raise Exception('Error: Not enough positive eigenvalues for the selected dim.')
	if w.size==0:
		raise Exception('Error: matrix is negative definite.')

	V = evecs[:,w]
	L = V.dot(np.diag(np.sqrt(evals[w]))).T
	N = D.shape[1]
	Lh = V.dot(np.diag(1./np.sqrt(evals[w]))).T
	Dm = D - np.tile(np.mean(Dl,axis=1),(N, 1)).T
	dim = w.size
	X = -Lh.dot(Dm)/2.
	X -= np.tile(np.mean(X,axis=1),(N, 1)).T

	_, evecs = sp.linalg.eigh(X.dot(X.T))

	return (evecs[:,::-1].T.dot(X)).T

def farthest_point_sample(N, dist, npoint):
    lands = np.arange(0, npoint)
    distance = np.ones((N)) * np.inf
    far_idx = np.random.randint(0, N)

    for i in range(0, npoint):
        lands[i] = far_idx
        d = dist[lands[i], :]               # 计算点集中的所有点到这个最远点的欧式距离
        mask = d < distance
        distance[mask] = d[mask]            # 更新distance，记录样本中每个点距离所有已出现的采样点的最小距离
        far_idx = np.argmax(distance, -1)   # 返回最远点索引

    return lands

def fix_connected_components(
    X,
    graph,
    n_connected_components,
    component_labels,
    mode="distance",
    metric="euclidean",
    **kwargs,
):
    if metric == "precomputed" and sp.sparse.issparse(X):
        raise RuntimeError(
            "fix_connected_components with metric='precomputed' requires the "
            "full distance matrix in X, and does not work with a sparse "
            "neighbors graph."
        )

    for i in range(n_connected_components):
        idx_i = np.flatnonzero(component_labels == i)
        Xi = X[idx_i]
        for j in range(i):
            idx_j = np.flatnonzero(component_labels == j)
            Xj = X[idx_j]

            if metric == "precomputed":
                D = X[np.ix_(idx_i, idx_j)]
            else:
                D = pairwise_distances(Xi, Xj, metric=metric, **kwargs)

            ii, jj = np.unravel_index(D.argmin(axis=None), D.shape)
            if mode == "connectivity":
                graph[idx_i[ii], idx_j[jj]] = 1
                graph[idx_j[jj], idx_i[ii]] = 1
            elif mode == "distance":
                graph[idx_i[ii], idx_j[jj]] = D[ii, jj]
                graph[idx_j[jj], idx_i[ii]] = D[ii, jj]
            else:
                raise ValueError(
                    "Unknown mode=%r, should be one of ['connectivity', 'distance']."
                    % mode
                )

    return graph

def geodesic_distance(X, k=10):
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(X)
    nbg = kneighbors_graph(nbrs, n_neighbors=k, n_jobs=-1)

    n_connected_components, labels = connected_components(nbg)
    if n_connected_components > 1:
        nbg = fix_connected_components(
            X=nbrs._fit_X,
            graph=nbg,
            n_connected_components=n_connected_components,
            component_labels=labels,
            mode="distance",
            metric=nbrs.effective_metric_,
            **nbrs.effective_metric_params_,
        )

    return shortest_path(nbg, directed=False)

class LandmarkMDS:

	def __init__(self, n_components=2):
		self.n_components = n_components

	def fit(self, X):
		self.X_train = X
		self.std = StandardScaler().fit(X)
		X = self.std.transform(X)
		D = cdist(X, X, 'euclidean')
		self.y_train = MDS(D, dim=self.n_components)

		return self

	def transform(self):
		if not hasattr(self, 'y_train'):
			raise Exception('Not fit yet.')
		return self.y_train
		
	def transform_oos(self, X_test, n_lands=None):
		X_test = self.std.transform(X_test)

		Dt = cdist(X_test, X_test, 'euclidean')

		lands_test = farthest_point_sample(X_test.shape[0], Dt, int(X_test.shape[0] / 10) if n_lands is None else n_lands)
		self.n_land = lands_test.shape[0]

		Dl2 = Dt[lands_test, :]

		return landmark_MDS(Dl2, lands_test, dim=self.n_components)
	
	def n_landmark(self):
		if not hasattr(self, 'n_land'):
			raise Exception('Not fit yet.')
		return self.n_land

class LandmarkIsomap:
	
	def __init__(self, n_neighbors=5, n_components=2):
		self.n_neighbors = n_neighbors
		self.n_components = n_components

	def fit(self, X):
		self.X_train = X
		self.std = StandardScaler().fit(X)
		X = self.std.transform(X)
		D = geodesic_distance(X, k=self.n_neighbors)
		self.y_train = MDS(D, dim=self.n_components)

		return self

	def transform(self):
		if not hasattr(self, 'y_train'):
			raise Exception('Not fit yet.')
		return self.y_train
		
	def transform_oos(self, X_test, n_lands=None):
		X_test = self.std.transform(X_test)

		Dt = geodesic_distance(X_test, k=self.n_neighbors)

		lands_test = farthest_point_sample(X_test.shape[0], Dt, int(X_test.shape[0] / 10) if n_lands is None else n_lands)
		self.n_land = lands_test.shape[0]

		Dl2 = Dt[lands_test, :]

		return landmark_MDS(Dl2, lands_test, dim=self.n_components)

	def n_landmark(self):
		if not hasattr(self, 'n_land'):
			raise Exception('Not fit yet.')
		return self.n_land