from scipy.io import loadmat
import numpy as np

def make_freyface():
    org_data = loadmat(f'org_datasets/frey_rawface.mat')

    print(org_data.keys())

    img = org_data['ff']
    X = np.transpose(img)

    print('X shape: ', X.shape)
    X = X.reshape((X.shape[0], 28, 20))

    print('X shape: ', X.shape)

    # projection using pca to 2d
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.reshape((X.shape[0], -1)))


    import matplotlib.pyplot as plt
    plt.scatter(X_pca[:, 0], X_pca[:, 1])
    plt.show()

    return X

if __name__ == '__main__':
    make_freyface()