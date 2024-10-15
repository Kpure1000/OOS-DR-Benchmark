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