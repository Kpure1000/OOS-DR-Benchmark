import numpy as np
import pandas as pd
import h5py as h5
import torch
from tqdm import tqdm

def save_projection(path, proj, label=None, **kwargs):
    df = pd.DataFrame()
    df['x']= proj[:,0]
    df['y']= proj[:,1]
    if label is not None:
        df['label'] = label
    for k, v in kwargs.items():
        df[k] = v
    df.to_csv(path, index=False)

def load_projection(path):
    df = pd.read_csv(path)
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    return np.concatenate((x[:, np.newaxis], y[:, np.newaxis]), axis=1)

def save_raw_data(path, X, **kwargs):
    '''
    Save raw data to hdf5 file
    
    Parameters
    ----------
    path: str
        path to save file
    X: np.ndarray
        raw data
    kwargs: dict
        additional data to save

    Example
    -------
    >>> save_raw_data('data.h5', X, y=labels)
    keys: [X, y]
    '''

    with h5.File(path, 'w') as f:
        f.create_dataset('X', data=X)
        for k, v in kwargs.items():
            f.create_dataset(k, data=v)

def save_dataset_any(path, X_train, X_test, **kewargs):
    with h5.File(path, 'w') as f:
        gE = f.create_group('E')
        gO = f.create_group('O')
        gE.create_dataset('X0', data=X_train)
        gO.create_dataset('X0', data=X_test)
        for k, v in kewargs.items():
            [name, set] = k.split('_')
            if set == 'train':
                gE.create_dataset(f'{name}0', data=v)
            elif set == 'test':
                gO.create_dataset(f'{name}0', data=v)
    

def save_dataset(path, X_train, X_test, y_train, y_test):
    save_datasets(path, [(X_train, X_test, y_train, y_test)])

def save_datasets(path, data:list):
    with h5.File(path, 'w') as f:
        gE = f.create_group('E')
        gO = f.create_group('O')
        for i, (X_train, X_test, y_train, y_test) in enumerate(data):
            gE.create_dataset(f'X{i}', data=X_train)
            gO.create_dataset(f'X{i}', data=X_test)
            gE.create_dataset(f'y{i}', data=y_train)
            gO.create_dataset(f'y{i}', data=y_test)

def load_datasets(path, n_stages=1):
    datas = []
    with h5.File(path, 'r') as f:
        for i in range(n_stages):
            datas.append((
                np.array(f['E'][f'X{i}']),
                np.array(f['O'][f'X{i}']),
                np.array(f['E'][f'y{i}']),
                np.array(f['O'][f'y{i}']),
            ))
    return datas

def load_dataset(path):
    return load_datasets(path)[0]


def plot_projection(projects: list,
                    titles: list = None,
                    y_train=None,
                    y_test=None,
                    cmap=None,
                    save_path=None,
                    train_size=3,
                    test_size=4,
                    train_alpha=0.1,
                    test_alpha=0.5,
                    train_marker='o',
                    test_marker='o',
                    train_labeled=False):
    import matplotlib.pyplot as plt

    ncols = min(4, len(projects))
    nrows = int(np.ceil(len(projects) / ncols))

    titles = [f'Method {str(i+1)}' for i in range(len(projects))] if titles is None else titles

    fig, ax = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    for i, (ptrain, ptest) in enumerate(projects):
        r = i // ncols
        c = i % ncols
        axes = ax[r, c] if nrows > 1 else ax[i] if ncols > 1 else ax

        label_train = np.zeros(ptrain.shape[0]) if y_train is None else y_train
        label_test = np.zeros(ptest.shape[0]) if y_test is None else y_test

        cmap_train = cmap if cmap is not None else 'tab10' if len(np.unique(label_train)) <= 10 else 'tab20'
        cmap_test  = cmap if cmap is not None else 'tab10' if len(np.unique(label_test )) <= 10 else 'tab20'

        axes.scatter(ptrain[:,0], ptrain[:,1],
                     c=label_train if train_labeled else 'gray',
                     cmap=cmap_train if train_labeled else None,
                     marker=train_marker,
                     s=train_size, alpha=train_alpha)

        axes.scatter(ptest[:,0], ptest[:,1],
                     c=label_test,
                     cmap=cmap_test,
                     marker=test_marker,
                     s=test_size, alpha=test_alpha)

        axes.set_title(titles[i])
        axes.set_xticks([])
        axes.set_yticks([])

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=400)


def farthest_point_sample(npoint, vert:np.array, norm:np.array=None):
    centroids = _farthest_point_sample_torch(torch.tensor(vert), npoint)
    v = vert[centroids[0],:]
    if norm is not None:
        n = norm[centroids[0],:]
        return v, n
    else:
        return v

def _farthest_point_sample_torch(vert, npoint):

    """
    Input:
        vert: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    if torch.cuda.is_available():
        xyz = vert.to("cuda")
    else:
        xyz = vert

    while xyz.dim() < 3:
        xyz = xyz.unsqueeze(0)

    device = xyz.device
    B, N, C = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.float).to(device)     # 采样点矩阵（B, npoint）
    distance = torch.ones(B, N, dtype=torch.float).to(device) * 1e10     # 采样点到所有点距离（B, N）

    batch_indices = torch.arange(B, dtype=torch.long).to(device)        # batch_size 数组

    #farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # 随机选择一点

    barycenter = torch.mean(xyz, dim=1, keepdim=True) #[B, 1, 3]

    dist = torch.sum((xyz - barycenter) ** 2, -1, dtype=torch.float) # [B, N, 1]
    farthest = torch.max(dist,1)[1]                                     #将距离重心最远的点作为第一个点

    for i in tqdm(range(npoint)):
        centroids[:, i] = farthest                                      # 更新第i个最远点
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)        # 取出这个最远点的xyz坐标
        dist = torch.sum((xyz - centroid) ** 2, -1, dtype=torch.float)                     # 计算点集中的所有点到这个最远点的欧式距离
        mask = dist < distance
        distance[mask] = dist[mask]                                     # 更新distance，记录样本中每个点距离所有已出现的采样点的最小距离

        farthest = torch.max(distance, -1)[1]                           # 返回最远点索引

    return centroids.cpu().int().tolist()
