import h5py as hf
import matplotlib.axis
import matplotlib.figure
from scipy.io import loadmat, savemat
import glob
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
import pandas as pd
import numpy as np
from numpy.random import mtrand
import matplotlib
import matplotlib.pyplot as plt
from metrics import Metrics
from methods.methods import Methods

def make_imdb():
    pos_files = glob.glob("org_datasets/aclImdb/train/pos/*.txt")
    pos_comments = []

    neg_files = glob.glob("org_datasets/aclImdb/train/neg/*.txt")
    neg_comments = []

    for pf in tqdm(pos_files, desc="Reading Positive", ncols=100, unit="txt"):
        with open(pf, 'r', encoding='utf-8') as f:
            pos_comments.append(' '.join(f.readlines()))

    for nf in tqdm(neg_files, desc="Reading Negative", ncols=100, unit="txt"):
        with open(nf, 'r', encoding='utf-8') as f:
            neg_comments.append(' '.join(f.readlines()))

    comments = pos_comments + neg_comments
    y = np.zeros((len(comments),)).astype('uint8')
    y[:len(pos_comments)] = 1

    tfidf = TfidfVectorizer(strip_accents='ascii',
                            stop_words='english', max_features=700)
    X = tfidf.fit_transform(comments).todense()
    data, _, labels, _ = train_test_split(X, y, train_size=0.13, random_state=42, stratify=y)

    with hf.File('org_datasets/aclImdb/imdb_3250.h5', 'w') as hfile:
        hfile.create_dataset('x', data=data)
        hfile.create_dataset('y', data=labels)

def make_bank():
    df = pd.read_csv('org_datasets/bank/bank-additional/bank-additional-full.csv', sep=';')

    y = np.array(df['y'] == 'yes').astype('int32')
    X = np.array(pd.get_dummies(df.drop('y', axis=1), dtype=np.float64))

    x_train, _, y_train, _ = train_test_split(X, y, train_size=0.05, random_state=42, stratify=y)

    with hf.File(f'org_datasets/bank/bank_{len(x_train)}.h5', 'w') as hfile:
        hfile.create_dataset('x', data=x_train)
        hfile.create_dataset('y', data=y_train)

def make_cifar10():
    import torchvision.datasets as datasets
    cifar10 = datasets.cifar.CIFAR10(root='org_datasets/cifar10', train=True, download=True, transform=None)
    X = cifar10.data.astype('float64') / 255.0
    y = np.array(cifar10.targets).astype('int32')

    X_train, _, y_train, _ = train_test_split(X, y, train_size=0.07, random_state=42, stratify=y)
    X = X_train
    y = y_train

    X = X[:,:,:,1]

    data = X.reshape((-1, 32 * 32))
    labels = y.squeeze()

    with hf.File(f'org_datasets/cifar10/cifar10_{data.shape[0]}.h5', 'w') as hfile:
        hfile.create_dataset('x', data=data)
        hfile.create_dataset('y', data=labels)

    return data, labels

def make_iris(draw=False):
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data
    y = iris.target

    with hf.File(f'org_datasets/iris/iris_{X.shape[0]}.h5', 'w') as hfile:
        hfile.create_dataset('x', data=X)
        hfile.create_dataset('y', data=y)

    if draw:
        pca = PCA(n_components=3)
        proj = pca.fit_transform(X)

        fig = plt.figure()
        plot = fig.add_subplot(projection='3d')
        plot.set_xlim((-5,5))
        plot.set_ylim((-5,5))
        plot.set_zlim((-5,5))
        plot.set_autoscale_on(False)
        plot.scatter(proj[:, 0], proj[:, 1], proj[:, 2], s=6, c=y)
        plt.show()

    return X, y

def make_kleinbottle(draw=False):
    np.random.seed(0)
    generator = mtrand._rand

    n_samples =3000

    u = 2 * np.pi * generator.uniform(size=n_samples)
    v = 2 * np.pi * generator.uniform(size=n_samples)
    R=1
    r=1

    x = (R + r * np.cos(u)) * np.cos(v)
    y = (R + r * np.cos(u)) * np.sin(v)
    z = r * np.sin(u) * np.cos(0.5 * v)
    w = r * np.sin(u) * np.sin(0.5 * v)

    X = np.vstack((x, y, z, w))
    X = X.T

    if draw:
        proj = PCA(n_components=3).fit_transform(X)
        fig = plt.figure()
        plot = fig.add_subplot(projection='3d')
        # plot.set_xlim((-5,5))
        # plot.set_ylim((-5,5))
        # plot.set_zlim((-5,5))
        # plot.set_autoscale_on(False)
        plot.scatter(proj[:, 0], proj[:, 1], proj[:, 2], s=3)
        plt.show()

    return X, u

def make_mobius(draw=False):
    np.random.seed(0)
    generator = mtrand._rand

    n_samples = 700
    w = 0.5

    u = 2 * np.pi * generator.uniform(size=n_samples)
    v = 2 * generator.uniform(size=n_samples) - 1

    vdiv2 = w * v * 0.5
    udiv2 = u * 0.5

    x = (1 + vdiv2 * np.cos(udiv2)) * np.cos(u)
    y = (1 + vdiv2 * np.cos(udiv2)) * np.sin(u)
    z = vdiv2 * np.sin(udiv2)

    X = np.vstack((x, y, z))
    X = X.T

    if draw:
        fig = plt.figure()
        plot = fig.add_subplot(projection='3d')
        plot.set_xlim((-5,5))
        plot.set_ylim((-5,5))
        plot.set_zlim((-5,5))
        plot.set_autoscale_on(False)
        plot.scatter(X[:, 0], X[:, 1], X[:, 2], s=6, c=u)
        plt.show()

    return X, u

def make_cnae9():
    df = pd.read_csv("org_datasets/cnae9/CNAE-9.data", header=None)
    y = np.array(df[0])
    X = np.array(df.drop(0, axis=1))

    with hf.File(f'org_datasets/cnae9/cnae9_{X.shape[0]}.h5', 'w') as hfile:
        hfile.create_dataset('x', data=X)
        hfile.create_dataset('y', data=y)

def make_cylinder_sample(draw):
    np.random.seed(0)
    generator = mtrand._rand

    n_samples = 1000
    r = 1.0
    h = 1.0
    x = r * generator.uniform(size=n_samples)
    y = generator.uniform(size=n_samples)

def make_cylinder_distribution(draw):
    np.random.seed(0)
    generator = mtrand._rand

    n_samples = 1000
    r = 1.0
    h = 1.0

def make_fashionmnist():
    from torchvision.datasets import FashionMNIST
    import torch
    fmnist = FashionMNIST(root='org_datasets/fashionmnist', train=True, download=True, transform=None)

    X = np.array(fmnist.data)
    y = np.array(fmnist.targets).astype('int32')

    print(X.shape, y.shape)

    X_train, _, y_train, _ = train_test_split(X, y, train_size=0.2, random_state=42, stratify=y)

    data = X_train.reshape((-1, 28 * 28))
    labels = y_train.squeeze()

    with hf.File(f'org_datasets/fashionmnist/fashionmnist_{data.shape[0]}.h5', 'w') as hfile:
        hfile.create_dataset('x', data=data)
        hfile.create_dataset('y', data=labels)

    return data, labels


def make_facedata():
    org_data = loadmat(f'org_datasets/face_data/face_data.mat')
    images = np.transpose(org_data['images'])
    poses = np.transpose(org_data['poses'])
    lights = np.transpose(org_data['lights'])
    X = images
    y = np.hstack((poses, lights))

    reducer = PCA(n_components=512, random_state=1)
    X = reducer.fit_transform(X)

    with hf.File(f'org_datasets/face_data/face_data.h5', 'w') as hfile:
        hfile.create_dataset('x', data=X)
        hfile.create_dataset('y', data=y)

    return X, y


def make_sms():
    df = pd.read_csv('org_datasets/sms/SMSSpamCollection', sep='\t', header=None)
    tfidf = TfidfVectorizer(strip_accents='ascii', stop_words='english', max_features=500)

    y = np.array(df[0] == 'spam').astype('uint8')
    X = tfidf.fit_transform(list(df[1])).todense()

    X_train, _, y_train, _ = train_test_split(X, y, train_size=0.15, random_state=42, stratify=y)
    X = X_train
    y = y_train

    with hf.File(f'org_datasets/sms/sms_{X.shape[0]}.h5', 'w') as hfile:
        hfile.create_dataset('x', data=X)
        hfile.create_dataset('y', data=y)

    return X, y

def make_plane(draw=False):
    np.random.seed(0)
    generator = mtrand._rand

    n_samples = 2000
    n_dims = 5

    S = [int(n_samples / 3), int(n_samples / 3), 0]
    S[2] = n_samples - S[0] - S[1]

    T = np.array([
        np.concatenate((
            generator.normal(1.0 * i - 0.0, 0.2 , size=S[0]),
            generator.normal(1.5 * i - 1.0, 0.2 , size=S[1]),
            generator.normal(1.0 * i + 1.0, 0.2  , size=S[2]),
        )) for i in range(n_dims - 1)
    ])
    lb = [0 for i in range(S[0])] + [1 for i in range(S[1])] + [2 for i in range(S[2])]

    A = generator.uniform(size=n_dims + 1)
    X = []
    for i in range(n_dims):
        if i < n_dims - 1:
            X.append(T[i] - A[0] / A[i])
        else:
            X.append(A[0] / A[i] * (np.sum(T, axis=0) - 1))

    X = np.array(X)
    X = np.transpose(X)
    X += 0.05 * generator.standard_normal(size=X.shape)

    os.makedirs('org_datasets/plane/', exist_ok=True)

    with hf.File(f'org_datasets/plane/plane_{X.shape[0]}.h5', 'w') as hfile:
        hfile.create_dataset('x', data=X)
        hfile.create_dataset('y', data=T[0])

    if draw:
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)

        fig = plt.figure()
        plot = fig.add_subplot()
        plot.scatter(X[:,0], X[:,1], s=2, c=T[0])
        plt.show()


def make_hyperball(n_samples, n_dims=2, x_center=0.0, draw=False, seed = 0):
    '''
    x_center: default 0
    radius: 1
    '''
    np.random.seed(seed)
    generator = mtrand._rand

    n_samples = n_samples
    n_dims = n_dims

    X = np.array([generator.normal(0, 1, n_dims) for i in range(n_samples)])

    for i, x in enumerate(X):
        norm = np.maximum(0.0001, np.sum(x**2)**0.5)
        r = generator.uniform(0, 1)**(1 / n_dims)
        X[i] = r * x / norm
        X[i][0] += x_center

    if draw:
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)

        fig = plt.figure()
        plot = fig.add_subplot()
        plot.scatter(X[:,0], X[:,1], s=2)
        plt.show()

    return X

def make_hyperballs(draw=False):
    n_samples = 1000
    n_dims = 2

    BallGiven = make_hyperball(n_samples, n_dims=n_dims, seed=1)
    BallNew = make_hyperball(n_samples, n_dims=n_dims, seed=2)
    label = np.concatenate((np.zeros(n_samples), np.ones(n_samples)))

    distance = [0.1 + i * 0.7 for i in range(4)]

    offset = np.zeros(BallNew.shape)
    offset[:, 0] = 1.0
    BallsOut = [BallNew + offset * d for d in distance]

    # with hf.File(f'org_datasets/hyperballs/hyperballs_{n_samples}.h5', 'w') as hfile:
    #     hfile['0'] = BallGiven
    #     for i, ball in enumerate(BallsOut):
    #         hfile[f'{i+1}'] = ball
    #     hfile['label'] = label

    if draw:
        fig = plt.figure()
        plots = [fig.add_subplot(100 + 10 * len(BallsOut) + i + 1) for i, _ in enumerate(BallsOut)]

        minx, miny, maxx, maxy = np.inf, np.inf, -np.inf, -np.inf

        twoballs = []

        for i,ball in enumerate(BallsOut):
            twoball = np.concatenate((BallGiven, ball))
            pca = PCA(n_components=2)
            # twoball = pca.fit_transform(twoball)
            twoballs.append(twoball)

            minx = min(minx, np.min(twoball[:, 0]))
            maxx = max(maxx, np.max(twoball[:, 0]))
            miny = min(miny, np.min(twoball[:, 1]))
            maxy = max(maxy, np.max(twoball[:, 1]))

        if maxx - minx > maxy - miny:
            maxy = (maxy + miny + maxx - minx) / 2
            miny = maxy - (maxx - minx)
        else:
            maxx = (maxx + minx + maxy - miny) / 2
            minx = maxx - (maxy - miny)

        maxy += (maxy-miny) * 0.05
        miny -= (maxy-miny) * 0.05
        maxx += (maxx-minx) * 0.05
        minx -= (maxx-minx) * 0.05

        color_map = [(0.18, 0.46, 0.71), (0.93, 0.5, 0.21)]
        c = np.array([color_map[int(l)] for l in label])

        for i, twoball in enumerate(twoballs):
            plots[i].set_xticks([])
            plots[i].set_yticks([])
            plots[i].set_xlim((minx, maxx))
            plots[i].set_ylim((miny, maxy))
            plots[i].set_aspect(1.0)
            plots[i].scatter(twoball[:,0], twoball[:,1], s=2, c=c)

        plt.show()

        for i,p in enumerate(plots):
            save_subfig(fig, plots[i], f'org_datasets/hyperballs/{i}.png', dpi=300)
            save_subfig(fig, plots[i], f'org_datasets/hyperballs/{i}.svg')


def save_subfig(fig: matplotlib.figure.Figure, ax: matplotlib.axis.Axis, file_name, dpi='figure'):
    bbox = ax.get_tightbbox(fig.canvas.get_renderer()).expanded(1., 1.)
    extent = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(file_name, bbox_inches=extent, dpi=dpi)

if __name__ == '__main__':
    # make_imdb()
    # make_bank()
    # data, labels = make_swiss_roll(True)
    # data, labels = make_cifar10()
    make_fashionmnist()
    # make_mobius()
    # make_iris(True)
    # make_cnae9()
    # make_facedata()
    # make_sms()
    # make_plane(True)
    # make_hyperballs(draw=True)

    # with hf.File(f'org_datasets/bank/bank_2059.h5', 'r') as hfile:
    #     X = np.array(hfile['x'][:])
    #     y = np.array(hfile['y'][:])

    # train_X, test_X, train_labels, test_labels = train_test_split(X, y, train_size=0.3, random_state=42, stratify=y)

    # data = {
    #     "train_X": train_X,
    #     "test_X": test_X,
    #     "train_labels": train_labels,
    #     "test_labels": test_labels
    # }
    # savemat("org_datasets/bank/bank.mat", data)
