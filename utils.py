import numpy as np
import pandas as pd

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