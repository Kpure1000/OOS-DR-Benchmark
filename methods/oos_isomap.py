import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors
from .landmark import geodesic_distance

class oos_Isomap:
    def __init__(self,n_components=2,n_neighbors=10):

        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.D = None
        self.n_samples = None
        self.n_features = None
        self.TD = None
        self.geodesic_TD = None
        self.M = None
        self.eigenvalues = None
        self.eigenvectors = None

    
    def get_geodesic_TD(self):
        # knn_graph = kneighbors_graph(self.D, n_neighbors=self.n_neighbors, mode='distance', include_self=True)
        # geodesic_TD = shortest_path(knn_graph, directed=False)

        geodesic_TD = geodesic_distance(self.D, k=self.n_neighbors)
        
        return geodesic_TD
    
    def get_M(self):
        
        tempM = self.geodesic_TD ** 2
        part1 = tempM
        part2 = -1.0 / self.n_samples * np.sum(tempM,axis=1).reshape((1,self.n_samples))
        part3 = -1.0 / self.n_samples * np.sum(tempM,axis=1).reshape((self.n_samples,1))
        part4 = 1.0 / self.n_samples ** 2 * np.sum(tempM)
        print("part4", np.unique(np.isfinite(self.geodesic_TD), return_counts=True))
        
        return -0.5 * (part1 + part2 + part3 + part4)


    def fit(self,D):
        self.D = np.array(D)
        self.n_samples = self.D.shape[0]
        self.n_features = self.D.shape[1]
        self.TD = squareform(pdist(self.D, 'euclidean'), force='no', checks=True)
        self.geodesic_TD = self.get_geodesic_TD()
        self.M = self.get_M()
        
        eigenvalues,eigenvectors = np.linalg.eig(self.M)
        # 按照特征值的绝对值从大到小排序
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        
        return self


    def get_y_k(self,i,k):
        return self.eigenvalues[k]**0.5 * self.eigenvectors[i,k]

    
    def get_y_k_oos(self,x,k):
        
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='auto').fit(self.D)
        distances, indices = nbrs.kneighbors([x])
        distances = distances[0] #n_neighbors
        indices = indices[0] #n_neighbors

        d1 = distances #n_neighbors
        d2 = self.geodesic_TD[:,indices] # n_samples * n_neighbors
        oos_x_geodesic_TD = np.min(d1 + d2,axis=1) #n_samples
        
        
        part1 = np.mean(self.geodesic_TD ** 2, axis=1) #n
        part2 = - oos_x_geodesic_TD ** 2
        sum = np.sum((part1 + part2) * self.eigenvectors[:,k])
        
        return 0.5 / self.eigenvalues[k]**0.5 * sum


    def get_Y_k_oos(self,X,k): #批量输入版
        
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='auto').fit(self.D)
        distances, indices = nbrs.kneighbors(X)
        
        # distance: n_samples * n_neighbors
        # indices: n_samples * n_neighbors

        Y = []
        
        for i in range(X.shape[0]):
            _dist = distances[i]
            _ind = indices[i]

            d1 = _dist # n_samples * n_neighbors
            d2 = self.geodesic_TD[:,_ind] # n_samples * n_neighbors
            oos_x_geodesic_TD = np.min(d1 + d2,axis=1) #n_samples
            
            
            part1 = np.mean(self.geodesic_TD ** 2, axis=1) #n
            part2 = - oos_x_geodesic_TD ** 2
            sum = np.sum((part1 + part2) * self.eigenvectors[:,k])
        
            Y.append(0.5 / self.eigenvalues[k]**0.5 * sum)
        
        return np.array(Y)
        
        
    def transform(self):
        
        raw_embedding = []
        for i in range(self.n_samples):
            raw_embedding.append([self.get_y_k(i,0),self.get_y_k(i,1)])
        raw_embedding = np.array(raw_embedding)
        
        return np.real(raw_embedding)

    def transform_oos(self,oos_X):

        
        # oos_embedding = []
        # for i in range(oos_X.shape[0]):
        #     oos_embedding.append([self.get_y_k_oos(oos_X[i],0),self.get_y_k_oos(oos_X[i],1)])
        # oos_embedding = np.array(oos_embedding)

        oos_embedding = np.concatenate((self.get_Y_k_oos(oos_X,0)[:,np.newaxis],self.get_Y_k_oos(oos_X,1)[:,np.newaxis]),axis=1)
        
        return np.real(oos_embedding)
    

'''
输入数据：
    样本内数据集X_train，格式为ndarray
    样本外数据集X_test，格式为ndarray

## 训练
oos_Isomap_instance = oos_Isomap(n_components=2,n_neighbors=10).fit(X_train)

## 获取样本内投影
embedding = oos_Isomap_instance.transform()

## 获取样本外投影
embedding_oos = oos_Isomap_instance.transform_oos(X_test)

'''