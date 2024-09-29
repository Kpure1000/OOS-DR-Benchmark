from copy import deepcopy
import numpy as np
from sklearn.metrics import auc, pairwise_distances, roc_curve, silhouette_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import scipy.stats, scipy.spatial
import scipy.spatial.distance as dis
import torch

class Metrics:
    def __init__(self):
        self.metrics = {
            #abbr        name                        function                     params           
            't':         ('Trustworthiness',         self.trustworthiness,        {'k': 7}         ),
            'c':         ('Continuity',              self.continuity,             {'k': 7}         ),
            'nh':        ('Neighborhood_Hit',        self.neighborhood_hit,       {'k': 7}         ),
            # 'ale':       ('Average_Local_Error',     self.average_local_error,    {}               ),
            'lc':        ('LC_meta_criterion',       self.LC_meta_criterion,      {'k': 7}         ),

            # 'ns':        ('Normalized_Stress',       self.normalized_stress,      {}               ),
            'sd':        ('Shepard_Goodness',        self.shepard_goodness,       {}               ),
            'tp':        ('Topographic_Product',     self.topographic_product,    {}               ),

            'sc':        ('Silhouette_Coefficient',  self.silhouette_coefficient, {'seed': 1}      ),
            'dsc':       ('DiStance_Consistency',    self.distance_consistency,   {}               ),
            
            # 'auc':       ('Area_Under_Curve',        self.area_under_curve,       {'k': 7}         ),
            'acc_oos':   ('ACCuracy_test',           self.accuracy,               {'train':False}  ),
            'acc_e':     ('ACCuracy_train',          self.accuracy,               {'train': True}  ),
        }

    def available(self):
        return list(self.metrics.keys())

    def run_single(self, metric_name:str):
        if metric_name not in self.metrics:
            raise ValueError(f"Unknown Metric '{metric_name}'")
        met = self.metrics[metric_name]
        func = met[1]
        kwargs = met[2]
        return (func(**kwargs), met[0])

    def update_metrics(self, X_train: np.ndarray, X_train_Embedded: np.ndarray, X_test: np.ndarray, X_test_Embedded: np.ndarray, y_train: np.ndarray=None, y_test: np.ndarray = None):
        self.X_train = X_train
        self.X_test = X_test

        self.Proj_test = X_test_Embedded
        self.Proj_train = X_train_Embedded

        self.y_train = y_train
        self.y_test = y_test

        self.dist_H = pairwise_distances(self.X_test, metric='euclidean')
        self.dist_L = pairwise_distances(self.Proj_test, metric='euclidean')
        self.dist_H_l = scipy.spatial.distance.pdist(self.X_test, 'euclidean')
        self.dist_L_l = scipy.spatial.distance.pdist(self.Proj_test, 'euclidean')

    def normalized_stress(self):
        return np.sum((self.dist_H_l - self.dist_L_l)**2) / np.sum(self.dist_H_l**2)

    def trustworthiness(self, k: int = 7):

        dist_H = self.dist_H.copy()
        np.fill_diagonal(dist_H, np.inf)
        ind_X = np.argsort(dist_H, axis=1)

        ind_E = (
            NearestNeighbors(n_neighbors=k)
            .fit(self.Proj_test)
            .kneighbors(return_distance=False)
        )

        n_samples = self.X_test.shape[0]

        inverted_index = np.zeros((n_samples, n_samples), dtype=int)
        ordered_indices = np.arange(n_samples + 1)
        inverted_index[ordered_indices[:-1, np.newaxis], ind_X] = ordered_indices[1:]
        ranks = (
            inverted_index[ordered_indices[:-1, np.newaxis], ind_E] - k
        )
        t = np.sum(ranks[ranks > 0])
        t = 1.0 - t * (
            2.0 / (n_samples * k * (2.0 * n_samples - 3.0 * k - 1.0))
        )
        return t

    def continuity(self, k: int = 7):
        ind_X = (
            NearestNeighbors(n_neighbors=k)
            .fit(self.X_test)
            .kneighbors(return_distance=False)
        )

        dist_L = self.dist_L.copy()
        np.fill_diagonal(dist_L, np.inf)
        ind_E = np.argsort(dist_L, axis=1)

        n_samples = self.X_test.shape[0]

        inverted_index = np.zeros((n_samples, n_samples), dtype=int)
        ordered_indices = np.arange(n_samples + 1)
        inverted_index[ordered_indices[:-1, np.newaxis], ind_E] = ordered_indices[1:]
        ranks = (
            inverted_index[ordered_indices[:-1, np.newaxis], ind_X] - k
        )
        c = np.sum(ranks[ranks > 0])
        c = 1.0 - c * (
            2.0 / (n_samples * k * (2.0 * n_samples - 3.0 * k - 1.0))
        )
        return c

    def shepard_goodness(self):
        return scipy.stats.spearmanr(a=self.dist_H_l, b=self.dist_L_l)[0]

    def silhouette_coefficient(self, seed: int = 42):
        if self.y_test is None:
            raise ValueError("Label is None")
        dist_E = self.dist_L.copy()
        dist_E[dist_E < 1e-7] = 0.0
        dist_E[dist_E == np.inf] = 0.0
        return silhouette_score(dist_E, self.y_test, random_state=seed, metric='precomputed')

    def distance_consistency(self):
        """
        Faster Distance Consistency (DSC)

        Returns
        --------
        [0, <b>1</b>]

        References
        --------
        [1] `Selecting good views of high-dimensional data using class consistency` (https://onlinelibrary.wiley.com/doi/10.1111/j.1467-8659.2009.01467.x)

        [2] https://github.com/hj-n/ltnc/blob/master/src/ltnc/cvm.py
        """
        if self.y_test is None:
            raise ValueError("Label is None")

        unique_labels = np.unique(self.y_test)
        n_samples = self.Proj_test.shape[0]
        n_centroids = unique_labels.shape[0]

        ## map labels to [0, n_centroids - 1]
        y = np.array(self.y_test)
        for i in range(len(unique_labels)):
            y[y == unique_labels[i]] = i

        ## get centroids
        centroids = []
        for i in range(n_centroids):
            centroids.append(np.mean(self.Proj_test[y == i], axis = 0))

        ## convert centroids to a column vector
        centroids = np.array(centroids)[:, np.newaxis]
        ## convert embedded to a row vector
        X_Embedded = np.array(self.Proj_test)[np.newaxis, :]

        ## get squared Centroid Distance mat, shape is (n_centroids, n_samples, 1)
        v_CX = (centroids - X_Embedded)
        sq_CD = np.einsum("ijk, ijk -> ij", v_CX, v_CX) # dot each element itself

        ## get the labels of the minimal squared distance
        min_CD_labels = np.argmin(sq_CD, axis=0).astype(y.dtype)

        ## count the correct points
        counts = np.sum(np.equal(min_CD_labels, y))

        return counts / n_samples

    def neighborhood_hit(self, k=7):
        if self.y_test is None:
            raise ValueError("Label is None")

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(self.Proj_test, self.y_test)

        neighbors = knn.kneighbors(self.Proj_test, return_distance=False)
        return np.mean(np.mean((self.y_test[neighbors] == np.tile(self.y_test.reshape((-1, 1)), k)).astype('uint8'), axis=1))

    # def area_under_curve(self, k=7):
    #     dist_mat = dis.squareform(dis.pdist(np.vstack((self.X_Embedded, self.X_org_Embedded))))
    #     n_samples = self.X_Embedded.shape[0]
    #     dist_mat_test2train = dist_mat[n_samples:, 0:n_samples]
    #     dist_mat_test2train = dist_mat_test2train.T
    #     dist_mat_test2train.sort(axis=1)
    #     y_socre = dist_mat_test2train[:, :k].sum(axis=1)
    #     y_true = (self.label > 0).astype(int)
    #     fpr, tpr, thresholds = roc_curve(y_true, y_socre, pos_label=1)
    #     return auc(fpr, tpr)


    def area_under_curve(self, k=7):
        if self.y_test is None:
            raise ValueError("Label is None")

        # 获取测试集样本到训练集样本的距离
        dist_mat = dis.cdist(self.Proj_test, self.Proj_train, metric='euclidean')
        dist_mat.sort(axis=1)
        # dist_mat = dis.squareform(dis.pdist(np.vstack((self.X_Embedded, self.X_org_Embedded))))

        # 获取测试集样本到训练集样本的距离
        # n_train = self.X_org_Embedded.shape[0]
        # n_test = self.X_Embedded.shape[0]
        # test_distances = dist_mat[n_test:, :n_train]

        # 计算每个测试样本到最近的k个训练样本的距离和
        # D_test = np.sum(test_distances[:, :k], axis=1)
        D_test = np.sum(dist_mat[:, :k], axis=1)

        # 将距离和转换为预测概率（这里使用简单的阈值方法）
        y_true = (self.y_test > 0).astype(int)
        # 计算AUC
        fpr, tpr, thresholds = roc_curve(y_true, D_test, pos_label=1)
        auc_score = auc(fpr, tpr)

        return auc_score


    def average_local_error(self):
        dist_H_Norm = self.dist_H / np.max(self.dist_H)
        dist_L_Norm = self.dist_L / np.max(self.dist_L)

        delta = np.abs(dist_H_Norm - dist_L_Norm)
        dia = delta.diagonal()

        mean_i = (delta.sum() - dia.sum()) / (self.Proj_test.shape[0] - 1)
        mean_ij = mean_i / self.Proj_test.shape[0]

        return mean_ij

    def accuracy(self, train: bool):

        proj = self.Proj_train if train else self.Proj_test
        y_true = self.y_train if train else self.y_test

        knn = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(proj)
        _, indices = knn.kneighbors(proj)

        y_pred = y_true[indices[:, 1]]

        return accuracy_score(y_true, y_pred)

    def topographic_product(self, batch_size=300):

        N = self.X_test.shape[0]
        k = N - 1
        EPS = 1e-7

        D_high = self.dist_H
        D_low = self.dist_L
        idx_high = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(self.X_test).kneighbors(self.X_test, n_neighbors=k, return_distance=False)
        idx_low = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(self.Proj_test).kneighbors(self.Proj_test, n_neighbors=k, return_distance=False)

        idx = np.arange(N)[1:, np.newaxis]
        idx_high_e1 = idx_high[1:, 1:k]
        idx_low_e1 = idx_low[1:, 1:k]

        Q1_j = (D_high[idx, idx_low_e1] + EPS) / (D_high[idx, idx_high_e1] + EPS)
        Q2_j = (D_low[idx, idx_low_e1] + EPS) / (D_low[idx, idx_high_e1] + EPS)

        P = 0

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            batch_idx = torch.arange(i, end, device=device)
            n = batch_idx.shape[0]

            idx_n_1 = torch.arange(start=i, end=end - 1, device=device)
            idx_b_n_1 = torch.arange(start=0, end=n - 1, device=device)
            idx_k_1 = torch.arange(k - 1, device=device)
                
            Q1_j_b = torch.tensor(Q1_j[idx_n_1.cpu()], dtype=torch.float64, device=device)
            Q2_j_b = torch.tensor(Q2_j[idx_n_1.cpu()], dtype=torch.float64, device=device)

            Q3_batch = torch.ones((n, k - 1), dtype=torch.float64, device=device)
            P3_batch = torch.ones((n, k - 1), dtype=torch.float64, device=device)

            # using log to avoid overflow
            log_Q1_j = torch.log(Q1_j_b[idx_b_n_1, :])
            log_Q2_j = torch.log(Q2_j_b[idx_b_n_1, :])
        
            for j in range(k - 1):
                # Q3_batch[idx_b_n_1, j] = torch.prod(Q1_j_b[idx_b_n_1, :j], dim=1) * torch.prod(Q2_j_b[idx_b_n_1, :j], dim=1)
                Q3_batch[idx_b_n_1, j] = torch.sum(log_Q1_j[idx_b_n_1, :j] + log_Q2_j[idx_b_n_1, :j], dim=1)
                
            # P3_batch[idx_b_n_1, :] = torch.pow(Q3_batch[idx_b_n_1, :], 1 / (2 * (idx_k_1 + 1)))
            P3_batch[idx_b_n_1, :] = torch.exp(Q3_batch[idx_b_n_1, :] / (2 * (idx_k_1 + 1)))

            P += torch.sum(torch.log(P3_batch)).cpu().item() / (N * (N - 1))

        return P

    def LC_meta_criterion(self, k=7):
        N = self.X_test.shape[0]
        idx_high = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(self.X_test).kneighbors(self.X_test, return_distance=False)
        idx_low = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(self.Proj_test).kneighbors(self.Proj_test, return_distance=False)

        Nk = 0
        for i in range(N):
            Nk += np.intersect1d(idx_high[i], idx_low[i]).shape[0]

        return Nk / N / (k)
