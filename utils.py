import numpy as np
import pandas as pd
import h5py as h5

def save_projection(path, proj, label=None):
    df = pd.DataFrame()
    df['x']= proj[:,0]
    df['y']= proj[:,1]
    if label is not None:
        df['label'] = label
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

def plot_projection(projects:list, titles:list=None, y_train=None, y_test=None, cmap=None, save_path=None, train_size=3, test_size=4, train_alpha=0.1, test_alpha=0.5, train_labeled=False):
    import matplotlib.pyplot as plt

    ncols = min(4, len(projects))
    nrows = int(np.ceil(len(projects) / ncols))

    titles = [f'Method {str(i+1)}' for i in range(len(projects))] if titles is None else titles

    fig, ax = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    for i, (ptrain, ptest) in enumerate(projects):
        r = i // nrows
        c = i % ncols
        axes = ax[r, c] if nrows > 1 else ax[i] if ncols > 1 else ax

        label_train = np.zeros(ptrain.shape[0]) if y_train is None else y_train
        label_test = np.zeros(ptest.shape[0]) if y_test is None else y_test

        cmap_train = cmap if cmap is not None else 'tab10' if len(np.unique(label_train)) <= 10 else 'tab20'
        cmap_test  = cmap if cmap is not None else 'tab10' if len(np.unique(label_test )) <= 10 else 'tab20'

        axes.scatter(ptrain[:,0], ptrain[:,1],
                     c=label_train if train_labeled else 'gray',
                     cmap=cmap_train if train_labeled else None,
                     s=train_size, alpha=train_alpha)

        axes.scatter(ptest[:,0], ptest[:,1],
                     c=label_test,
                     cmap=cmap_test,
                     s=test_size, alpha=test_alpha)

        axes.set_title(titles[i])
        axes.set_xticks([])
        axes.set_yticks([])

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=400)
