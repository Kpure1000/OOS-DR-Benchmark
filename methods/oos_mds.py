import numpy as np
from scipy.spatial.distance import pdist, squareform

class oos_MDS:
    def __init__(self,n_components=2):

        self.n_components = n_components
        self.D = None
        self.n_samples = None
        self.n_features = None
        self.TD = None
        self.M = None
        self.eigenvalues = None
        self.eigenvectors = None
    


    def get_M(self):
        
        part1 = self.TD ** 2
        part2 = -np.mean(self.TD ** 2,axis=0).reshape((1,self.n_samples))
        part3 = -np.mean(self.TD ** 2,axis=1).reshape((self.n_samples,1))
        part4 = np.mean(self.TD ** 2)

        return -0.5 * (part1 + part2 + part3 + part4)


    def fit(self,D):
        self.D = np.array(D)
        self.n_samples = self.D.shape[0]
        self.n_features = self.D.shape[1]
        self.TD = squareform(pdist(self.D, 'euclidean'), force='no', checks=True)
        self.M = self.get_M()

        eigenvalues,eigenvectors = np.linalg.eig(self.M)
        # 按照特征值的绝对值从大到小排序
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        
        return self


    def getTD_single_and_multi(self,single,multi):

        return np.linalg.norm(single - multi,ord=2,axis=1)

    def mds_kernel(self,a,b):

        part1 = np.linalg.norm(a - b,ord=2) ** 2
        part2 = -np.mean(self.getTD_single_and_multi(single=a,multi=self.D) ** 2)
        part3 = -np.mean(self.getTD_single_and_multi(single=b,multi=self.D) ** 2)
        part4 = np.mean(self.TD ** 2)

        return -0.5 * (part1 + part2 + part3 + part4)


    def get_y_k(self,i,k):
        return self.eigenvalues[k]**0.5 * self.eigenvectors[i,k]

    def get_y_k_oos(self,x,k):
        
        oos_x_TD = self.getTD_single_and_multi(x,self.D) # 样本外的点距离D个点的距离 n array
        
        part1 = oos_x_TD ** 2 # n
        part2 = -np.mean(oos_x_TD ** 2) #1
        part3 = -np.mean(self.TD ** 2,axis=1) #n
        part4 = np.mean(self.TD ** 2) #1
        mds_kernels = -0.5 * (part1 + part2 + part3 + part4)
        sum = np.sum(mds_kernels * self.eigenvectors[:,k])
        
        return 1.0 / self.eigenvalues[k]**0.5 * sum
    
    def transform(self):
        '''
        return the raw embedding
        '''
        raw_embedding = []
        for i in range(self.n_samples):
            raw_embedding.append([self.get_y_k(i,0),self.get_y_k(i,1)])
        raw_embedding = np.array(raw_embedding)
        
        return np.real(raw_embedding)

    def transform_oos(self,oos_X):
        '''
        return the oos embedding
        '''
        oos_embedding = []
        for i in range(oos_X.shape[0]):
            oos_embedding.append([self.get_y_k_oos(oos_X[i],0),self.get_y_k_oos(oos_X[i],1)])
        oos_embedding = np.array(oos_embedding)
        
        return np.real(oos_embedding)
    

'''

输入数据：
    样本内数据集X_train，格式为ndarray
    样本外数据集X_test，格式为ndarray

## 训练
oos_MDS_instance = oos_MDS(n_components=2).fit(X_train)

## 获取样本内投影
embedding = oos_MDS_instance.transform()

## 获取样本外投影
embedding_oos = oos_MDS_instance.transform_oos(X_test)

'''