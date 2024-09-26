from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

import h5py as hf
import numpy as np
from numpy.random import mtrand

def make_plane_structure(path, n_samples):
    np.random.seed(0)
    generator = mtrand._rand

    o = np.array([0.0, 0.0, 0.0])
    n = np.array([0.5, -0.5, 0.5])

    def plane_eq(x, y):
        return -(n[0] * (x - o[0]) + n[1] * (y - o[1])) / n[2] + o[2]

    # get point on plane
    p = np.array([1.0, 1.0, 0.0])
    p[2] = plane_eq(p[0], p[1])

    # get orthogonal basis
    p = p / np.linalg.norm(p)
    q = np.cross(n, p)
    q = q / np.linalg.norm(q)

    # using orthogonal basis to generate points on plane

    with hf.File(path, 'w') as f:
        gE = f.create_group('E')
        gO = f.create_group('O')

        n_samples_train = int(n_samples * 0.7)
        n_samples_test = n_samples - n_samples_train
        n_sub_train = n_samples_train // 3
        n_sub_test = n_samples_test // 3

        y_train = np.concatenate((np.zeros(n_sub_train), np.ones(n_sub_train), 2*np.ones(n_samples_train-2*n_sub_train)))
        y_test = np.concatenate((np.zeros(n_sub_test), np.ones(n_sub_test), 2*np.ones(n_samples_test-2*n_sub_test)))
        gE.create_dataset(f'y{0}', shape=y_train.shape, dtype=y_train.dtype, data=y_train)
        gO.create_dataset(f'y{0}', shape=y_test.shape, dtype=y_test.dtype, data=y_test)

        sigma = 0.15

        # train set p
        train_set_p1 = generator.normal(0.0, sigma, size=n_sub_train) # class 0
        train_set_p2 = generator.normal(0.5, sigma, size=n_sub_train) # class 1
        train_set_p3 = generator.normal(1.0, sigma, size=n_samples_train-2*n_sub_train) # class 2
        train_set_p = np.concatenate((train_set_p1, train_set_p2, train_set_p3))
        # train set q
        train_set_q1 = generator.normal(0.0, sigma, size=n_sub_train) # class 0
        train_set_q2 = generator.normal(0.5, sigma, size=n_sub_train) # class 1
        train_set_q3 = generator.normal(1.0, sigma, size=n_samples_train-2*n_sub_train) # class 2
        train_set_q = np.concatenate((train_set_q3, train_set_q1, train_set_q2))

        X_train = train_set_p[:, np.newaxis] * p + train_set_q[:, np.newaxis] * q
        X_train += 0.03 * generator.standard_normal(size=X_train.shape)

        # test set p
        test_set_p1 = generator.normal(0.0, sigma, size=n_sub_test) # class 0
        test_set_p2 = generator.normal(0.5, sigma, size=n_sub_test) # class 1
        test_set_p3 = generator.normal(1.0, sigma, size=n_samples_test-2*n_sub_test) # class 2
        test_set_p = np.concatenate((test_set_p1, test_set_p2, test_set_p3))
        # test set q
        test_set_q1 = generator.normal(0.0, sigma, size=n_sub_test) # class 1
        test_set_q2 = generator.normal(0.5, sigma, size=n_sub_test)  # class 0
        test_set_q3 = generator.normal(1.0, sigma, size=n_samples_test-2*n_sub_test) # class 2
        test_set_q = np.concatenate((test_set_q3, test_set_q1, test_set_q2))

        X_test = test_set_p[:, np.newaxis] * p + test_set_q[:, np.newaxis] * q
        X_test += 0.03 * generator.standard_normal(size=X_test.shape)

        gE.create_dataset(f'X{0}', shape=X_train.shape, dtype=X_train.dtype, data=X_train)
        gO.create_dataset(f'X{0}', shape=X_test.shape, dtype=X_test.dtype, data=X_test)


def make_swissroll_structure(path, n_samples, width = 1.0, k = 1.0, u = 1.5):
    np.random.seed(0)
    generator = mtrand._rand

    t_max = 2.0 * np.pi

    def swiss_roll(n_samples, center1, center2, offset1=0, offset2=0):
        t = t_max * generator.normal(center1 + offset1, 0.07, size=n_samples)
        t = np.sort(t)
        y = width * generator.normal(center2 + offset2, 0.1, size=n_samples)
        x = k * t * np.cos(u * t)
        z = k * t * np.sin(u * t)
        X = np.vstack((x, y, z))
        X += 0.1 * generator.standard_normal(size=X.shape)
        X = X.T

        return X

    with hf.File(path, 'w') as f:
        gE = f.create_group('E')
        gO = f.create_group('O')

        n_samples_train = int(n_samples * 0.7)
        n_samples_test = n_samples - n_samples_train

        n_sub_train = n_samples_train // 3
        n_sub_test = n_samples_test // 3

        y_train = np.concatenate((np.zeros(n_sub_train), np.ones(n_sub_train), 2 * np.ones(n_samples_train - 2 * n_sub_train)))
        y_test = np.concatenate((np.zeros(n_sub_test), np.ones(n_sub_test), 2*np.ones(n_samples_test-2*n_sub_test)))

        gE.create_dataset(f'y{0}', shape=y_train.shape, dtype=y_train.dtype, data=y_train)
        gO.create_dataset(f'y{0}', shape=y_test.shape, dtype=y_test.dtype, data=y_test)

        X_train = np.concatenate((
            swiss_roll(n_sub_train, 0.2, 0.2),
            swiss_roll(n_sub_train, 0.5, 0.3),
            swiss_roll(n_samples_train - 2 * n_sub_train, 0.8, 0.2)
        ))
        X_test = np.concatenate((
            swiss_roll(n_sub_test, 0.2, 0.2),
            swiss_roll(n_sub_test, 0.5, 0.3),
            swiss_roll(n_samples_test - 2 * n_sub_test, 0.8, 0.2)
        ))

        gE.create_dataset(f'X{0}', shape=X_train.shape, dtype=X_train.dtype, data=X_train)
        gO.create_dataset(f'X{0}', shape=X_test.shape, dtype=X_test.dtype, data=X_test)


def make_hybrid_structure(path, n_samples, width = 0.5, k = 0.7, u = 2.0):
    np.random.seed(0)
    generator = mtrand._rand

    t_max = 2.0 * np.pi

    def swiss_roll(n_samples, center1, center2, offset1=0, offset2=0):
        t = t_max * generator.normal(center1 + offset1, 0.2, size=n_samples)
        t = np.sort(t)
        y = width * generator.normal(center2 + offset2, 0.6, size=n_samples)
        x = k * t * np.cos(u * t)
        z = k * t * np.sin(u * t)
        X = np.vstack((x, y, z))
        X = X.T

        return X

    o = np.array([-1.0, -1.0, 0.0])
    n = np.array([0.0, 1.0, 0.01])
    # n /= np.linalg.norm(n)

    def plane_eq(x, y):
        return -(n[0] * (x - o[0]) + n[1] * (y - o[1])) / n[2] + o[2]

    # get point on plane
    p = np.array([0.0, 0.0, 0.0])
    p[2] = plane_eq(p[0], p[1])

    # get orthogonal basis
    p = p / np.linalg.norm(p)
    q = np.cross(n, p)
    q = q / np.linalg.norm(q)

    with hf.File(path, 'w') as f:
        gE = f.create_group('E')
        gO = f.create_group('O')

        n_samples_train = int(n_samples * 0.7)
        n_samples_test = n_samples - n_samples_train
        n_sub_train = n_samples_train // 4
        n_sub_test = n_samples_test // 4

        y_train = np.concatenate(
            (np.zeros(n_sub_train), np.ones(n_sub_train),
             2 * np.ones(n_sub_train),
             3 * np.ones(n_samples_train- 3 * n_sub_train),
             ))
        y_test = np.concatenate((
            np.zeros(n_sub_test), 
            np.ones(n_sub_test),
            2 * np.ones(n_sub_test),
            3 * np.ones(n_samples_test - 3 * n_sub_test),
            ))

        gE.create_dataset(f'y{0}', shape=y_train.shape, dtype=y_train.dtype, data=y_train)
        gO.create_dataset(f'y{0}', shape=y_test.shape, dtype=y_test.dtype, data=y_test)

        sigma = 1.2
        train_set_p = generator.normal(0.0, sigma, size=n_sub_train)
        train_set_q = generator.normal(-0.5, sigma, size=n_sub_train)
        plane_train = train_set_p[:, np.newaxis] * p + train_set_q[:, np.newaxis] * q
        train_set_p1 = generator.normal(-0.8, sigma, size=n_samples_train - 3 * n_sub_train)
        train_set_q1 = generator.normal(1.3, sigma, size=n_samples_train - 3 * n_sub_train)
        plane_train1 = train_set_p1[:, np.newaxis] * p + train_set_q1[:, np.newaxis] * q
        X_train = np.concatenate((
            swiss_roll(n_sub_train, 0.3, 0.3),
            swiss_roll(n_sub_train, 0.4, 0.4),
            plane_train,
            plane_train1
        ))

        test_set_p = generator.normal(0.0, sigma, size=n_sub_test)
        test_set_q = generator.normal(-0.5, sigma, size=n_sub_test)
        plane_test = test_set_p[:, np.newaxis] * p + test_set_q[:, np.newaxis] * q
        test_set_p1 = generator.normal(0.8, sigma, size=n_samples_test - 3 * n_sub_test)
        test_set_q1 = generator.normal(1.3, sigma, size=n_samples_test - 3 * n_sub_test)
        plane_test1 = test_set_p1[:, np.newaxis] * p + test_set_q1[:, np.newaxis] * q
        X_test = np.concatenate((
            swiss_roll(n_sub_test, 0.3, 0.3),
            swiss_roll(n_sub_test, 0.4, 0.4),
            plane_test,
            plane_test1
        ))

        gE.create_dataset(f'X{0}', shape=X_train.shape, dtype=X_train.dtype, data=X_train)
        gO.create_dataset(f'X{0}', shape=X_test.shape, dtype=X_test.dtype, data=X_test)


# 比例差异
def make_plane_prop(path, n_samples, n_dims = 3):
    # 训练集和测试集的比例
    props = [
        (0.9, 0.1),
        (0.7, 0.3),
        (0.5, 0.5),
        (0.3, 0.7),
    ]

    np.random.seed(0)
    generator = mtrand._rand

    o = np.array([0.0, 0.0, 0.0])
    n = np.array([0.5, -0.5, 0.5])

    def plane_eq(x, y):
        return -(n[0] * (x - o[0]) + n[1] * (y - o[1])) / n[2] + o[2]

    # get point on plane
    p = np.array([1.0, 1.0, 0.0])
    p[2] = plane_eq(p[0], p[1])

    # get orthogonal basis
    p = p / np.linalg.norm(p)
    q = np.cross(n, p)
    q = q / np.linalg.norm(q)

    # using orthogonal basis to generate points on plane

    with hf.File(path, 'w') as f:
        gE = f.create_group('E')
        gO = f.create_group('O')

        for idx, prop in enumerate(props):

            n_samples_train = int(n_samples * prop[0])
            n_samples_test = n_samples - n_samples_train
            n_sub_train = n_samples_train // 3
            n_sub_test = n_samples_test // 3

            y_train = np.concatenate((np.zeros(n_sub_train), np.ones(n_sub_train), 2*np.ones(n_samples_train-2*n_sub_train)))
            y_test = np.concatenate((np.zeros(n_sub_test), np.ones(n_sub_test), 2*np.ones(n_samples_test-2*n_sub_test)))
            gE.create_dataset(f'y{idx}', shape=y_train.shape, dtype=y_train.dtype, data=y_train)
            gO.create_dataset(f'y{idx}', shape=y_test.shape, dtype=y_test.dtype, data=y_test)

            sigma = 0.15

            # train set p
            train_set_p1 = generator.normal(0.0, sigma, size=n_sub_train) # class 0
            train_set_p2 = generator.normal(0.5, sigma, size=n_sub_train) # class 1
            train_set_p3 = generator.normal(1.0, sigma, size=n_samples_train-2*n_sub_train) # class 2
            train_set_p = np.concatenate((train_set_p1, train_set_p2, train_set_p3))
            # train set q
            train_set_q1 = generator.normal(0.0, sigma, size=n_sub_train) # class 0
            train_set_q2 = generator.normal(0.5, sigma, size=n_sub_train) # class 1
            train_set_q3 = generator.normal(1.0, sigma, size=n_samples_train-2*n_sub_train) # class 2
            train_set_q = np.concatenate((train_set_q3, train_set_q1, train_set_q2))

            X_train = train_set_p[:, np.newaxis] * p + train_set_q[:, np.newaxis] * q
            X_train += 0.03 * generator.standard_normal(size=X_train.shape)

            # test set p
            test_set_p1 = generator.normal(0.0, sigma, size=n_sub_test) # class 0
            test_set_p2 = generator.normal(0.5, sigma, size=n_sub_test) # class 1
            test_set_p3 = generator.normal(1.0, sigma, size=n_samples_test-2*n_sub_test) # class 2
            test_set_p = np.concatenate((test_set_p1, test_set_p2, test_set_p3))
            # test set q
            test_set_q1 = generator.normal(0.0, sigma, size=n_sub_test) # class 1
            test_set_q2 = generator.normal(0.5, sigma, size=n_sub_test)  # class 0
            test_set_q3 = generator.normal(1.0, sigma, size=n_samples_test-2*n_sub_test) # class 2
            test_set_q = np.concatenate((test_set_q3, test_set_q1, test_set_q2))

            X_test = test_set_p[:, np.newaxis] * p + test_set_q[:, np.newaxis] * q
            X_test += 0.03 * generator.standard_normal(size=X_test.shape)

            gE.create_dataset(f'X{idx}', shape=X_train.shape, dtype=X_train.dtype, data=X_train)
            gO.create_dataset(f'X{idx}', shape=X_test.shape, dtype=X_test.dtype, data=X_test)


def make_swissroll_prop(path, n_samples, width = 1.0, k = 1.0, u = 1.5):
    np.random.seed(0)
    generator = mtrand._rand

    t_max = 2.0 * np.pi

    # 训练集和测试集的比例
    props = [
        (0.9, 0.1),
        (0.7, 0.3),
        (0.5, 0.5),
        (0.3, 0.7),
    ]

    def swiss_roll(n_samples, center1, center2, offset1=0, offset2=0):
        t = t_max * generator.normal(center1 + offset1, 0.07, size=n_samples)
        t = np.sort(t)
        y = width * generator.normal(center2 + offset2, 0.1, size=n_samples)
        x = k * t * np.cos(u * t)
        z = k * t * np.sin(u * t)
        X = np.vstack((x, y, z))
        X += 0.1 * generator.standard_normal(size=X.shape)
        X = X.T

        return X

    with hf.File(path, 'w') as f:
        gE = f.create_group('E')
        gO = f.create_group('O')

        for idx, prop in enumerate(props):
            n_samples_train = int(n_samples * prop[0])
            n_samples_test = n_samples - n_samples_train

            n_sub_train = n_samples_train // 3
            n_sub_test = n_samples_test // 3

            y_train = np.concatenate((np.zeros(n_sub_train), np.ones(n_sub_train), 2 * np.ones(n_samples_train - 2 * n_sub_train)))
            y_test = np.concatenate((np.zeros(n_sub_test), np.ones(n_sub_test), 2*np.ones(n_samples_test-2*n_sub_test)))
            gE.create_dataset(f'y{idx}', shape=y_train.shape, dtype=y_train.dtype, data=y_train)
            gO.create_dataset(f'y{idx}', shape=y_test.shape, dtype=y_test.dtype, data=y_test)

            X_train = np.concatenate((
                swiss_roll(n_sub_train, 0.2, 0.2),
                swiss_roll(n_sub_train, 0.5, 0.3),
                swiss_roll(n_samples_train - 2 * n_sub_train, 0.8, 0.2)
            ))
            X_test = np.concatenate((
                swiss_roll(n_sub_test, 0.2, 0.2),
                swiss_roll(n_sub_test, 0.5, 0.3),
                swiss_roll(n_samples_test - 2 * n_sub_test, 0.8, 0.2)
            ))

            gE.create_dataset(f'X{idx}', shape=X_train.shape, dtype=X_train.dtype, data=X_train)
            gO.create_dataset(f'X{idx}', shape=X_test.shape, dtype=X_test.dtype, data=X_test)


def make_hybrid_prop(path, n_samples, width = 0.5, k = 0.7, u = 2.0):
    # 训练集和测试集的比例
    props = [
        (0.9, 0.1),
        (0.7, 0.3),
        (0.5, 0.5),
        (0.3, 0.7),
    ]

    np.random.seed(0)
    generator = mtrand._rand

    t_max = 2.0 * np.pi

    def swiss_roll(n_samples, center1, center2, offset1=0, offset2=0):
        t = t_max * generator.normal(center1 + offset1, 0.2, size=n_samples)
        t = np.sort(t)
        y = width * generator.normal(center2 + offset2, 0.6, size=n_samples)
        x = k * t * np.cos(u * t)
        z = k * t * np.sin(u * t)
        X = np.vstack((x, y, z))
        X = X.T

        return X

    o = np.array([-1.0, -1.0, 0.0])
    n = np.array([0.0, 1.0, 0.01])
    # n /= np.linalg.norm(n)

    def plane_eq(x, y):
        return -(n[0] * (x - o[0]) + n[1] * (y - o[1])) / n[2] + o[2]

    # get point on plane
    p = np.array([0.0, 0.0, 0.0])
    p[2] = plane_eq(p[0], p[1])

    # get orthogonal basis
    p = p / np.linalg.norm(p)
    q = np.cross(n, p)
    q = q / np.linalg.norm(q)

    with hf.File(path, 'w') as f:
        gE = f.create_group('E')
        gO = f.create_group('O')

        for idx,prop in enumerate(props):
            n_samples_train = int(n_samples * prop[0])
            n_samples_test = n_samples - n_samples_train
            n_sub_train = n_samples_train // 4
            n_sub_test = n_samples_test // 4

            y_train = np.concatenate(
                (np.zeros(n_sub_train), np.ones(n_sub_train),
                2 * np.ones(n_sub_train),
                3 * np.ones(n_samples_train- 3 * n_sub_train),
                ))
            y_test = np.concatenate((
                np.zeros(n_sub_test), 
                np.ones(n_sub_test),
                2 * np.ones(n_sub_test),
                3 * np.ones(n_samples_test - 3 * n_sub_test),
                ))

            gE.create_dataset(f'y{idx}', shape=y_train.shape, dtype=y_train.dtype, data=y_train)
            gO.create_dataset(f'y{idx}', shape=y_test.shape, dtype=y_test.dtype, data=y_test)

            sigma = 1.2
            train_set_p = generator.normal(0.0, sigma, size=n_sub_train)
            train_set_q = generator.normal(-0.5, sigma, size=n_sub_train)
            plane_train = train_set_p[:, np.newaxis] * p + train_set_q[:, np.newaxis] * q
            train_set_p1 = generator.normal(-0.8, sigma, size=n_samples_train - 3 * n_sub_train)
            train_set_q1 = generator.normal(1.3, sigma, size=n_samples_train - 3 * n_sub_train)
            plane_train1 = train_set_p1[:, np.newaxis] * p + train_set_q1[:, np.newaxis] * q
            X_train = np.concatenate((
                swiss_roll(n_sub_train, 0.3, 0.3),
                swiss_roll(n_sub_train, 0.4, 0.4),
                plane_train,
                plane_train1
            ))

            test_set_p = generator.normal(0.0, sigma, size=n_sub_test)
            test_set_q = generator.normal(-0.5, sigma, size=n_sub_test)
            plane_test = test_set_p[:, np.newaxis] * p + test_set_q[:, np.newaxis] * q
            test_set_p1 = generator.normal(0.8, sigma, size=n_samples_test - 3 * n_sub_test)
            test_set_q1 = generator.normal(1.3, sigma, size=n_samples_test - 3 * n_sub_test)
            plane_test1 = test_set_p1[:, np.newaxis] * p + test_set_q1[:, np.newaxis] * q
            X_test = np.concatenate((
                swiss_roll(n_sub_test, 0.3, 0.3),
                swiss_roll(n_sub_test, 0.4, 0.4),
                plane_test,
                plane_test1
            ))

            gE.create_dataset(f'X{idx}', shape=X_train.shape, dtype=X_train.dtype, data=X_train)
            gO.create_dataset(f'X{idx}', shape=X_test.shape, dtype=X_test.dtype, data=X_test)



def make_plane_dist(path, n_samples):
    np.random.seed(0)
    generator = mtrand._rand

    o = np.array([0.0, 0.0, 0.0])
    n = np.array([0.5, -0.5, 0.5])

    def plane_eq(x, y):
        return -(n[0] * (x - o[0]) + n[1] * (y - o[1])) / n[2] + o[2]

    # get point on plane
    p = np.array([1.0, 1.0, 0.0])
    p[2] = plane_eq(p[0], p[1])

    # get orthogonal basis
    p = p / np.linalg.norm(p)
    q = np.cross(n, p)
    q = q / np.linalg.norm(q)

    # using orthogonal basis to generate points on plane

    dists = [(0.0, 0.0), (1.0, 1.0)]

    with hf.File(path, 'w') as f:
        gE = f.create_group('E')
        gO = f.create_group('O')

        n_samples_train = int(n_samples * 0.7)
        n_samples_test = n_samples - n_samples_train
        n_sub_train = n_samples_train // 3
        n_sub_test = n_samples_test // 2

        y_train = np.concatenate((np.zeros(n_sub_train), np.ones(n_sub_train), 2*np.ones(n_samples_train-2*n_sub_train)))
        y_test = np.concatenate((np.zeros(n_sub_test), np.ones(n_samples_test-n_sub_test)))

        for idx, dist in enumerate(dists):
            gE.create_dataset(f'y{idx}', shape=y_train.shape, dtype=y_train.dtype, data=y_train)
            gO.create_dataset(f'y{idx}', shape=y_test.shape, dtype=y_test.dtype, data=y_test)
            # train set p
            train_set_p1 = generator.normal(0.0, 0.15, size=n_sub_train) # class 0
            train_set_p2 = generator.normal(0.5, 0.15, size=n_sub_train) # class 1
            train_set_p3 = generator.normal(1.0, 0.15, size=n_samples_train-2*n_sub_train) # class 2
            train_set_p = np.concatenate((train_set_p1, train_set_p2, train_set_p3))
            # train set q
            train_set_q1 = generator.normal(0.0, 0.15, size=n_sub_train) # class 0
            train_set_q2 = generator.normal(0.5, 0.15, size=n_sub_train) # class 1
            train_set_q3 = generator.normal(1.0, 0.15, size=n_samples_train-2*n_sub_train) # class 2
            train_set_q = np.concatenate((train_set_q3, train_set_q1, train_set_q2))

            X_train = train_set_p[:, np.newaxis] * p + train_set_q[:, np.newaxis] * q
            X_train += 0.03 * generator.standard_normal(size=X_train.shape)

            # test set p
            test_set_p1 = generator.normal(0.0 + dist[0], 0.15, size=n_sub_test) # class 0
            test_set_p2 = generator.normal(1.0 + dist[0], 0.15, size=n_samples_test-n_sub_test) # class 1
            test_set_p = np.concatenate((test_set_p1, test_set_p2))
            # test set q
            test_set_q1 = generator.normal(1.0 + dist[1], 0.15, size=n_sub_test) # class 1
            test_set_q2 = generator.normal(0.0 + dist[1], 0.15, size=n_samples_test-n_sub_test)  # class 0
            test_set_q = np.concatenate((test_set_q1, test_set_q2))

            X_test = test_set_p[:, np.newaxis] * p + test_set_q[:, np.newaxis] * q
            X_test += 0.03 * generator.standard_normal(size=X_test.shape)

            gE.create_dataset(f'X{idx}', shape=X_train.shape, dtype=X_train.dtype, data=X_train)
            gO.create_dataset(f'X{idx}', shape=X_test.shape, dtype=X_test.dtype, data=X_test)


def make_swissroll_dist(path, n_samples, width = 1.0, k = 1.0, u = 1.5):
    np.random.seed(0)
    generator = mtrand._rand

    t_max = 2.0 * np.pi

    dists = [0.0, 0.8]

    def swiss_roll(n_samples, center1, center2, offset1=0, offset2=0):
        t = t_max * generator.normal(center1 + offset1, 0.07, size=n_samples)
        t = np.sort(t)
        y = width * generator.normal(center2 + offset2, 0.1, size=n_samples)
        x = k * t * np.cos(u * t)
        z = k * t * np.sin(u * t)
        X = np.vstack((x, y, z))
        X += 0.1 * generator.standard_normal(size=X.shape)
        X = X.T

        return X

    with hf.File(path, 'w') as f:
        gE = f.create_group('E')
        gO = f.create_group('O')

        n_samples_train = int(n_samples * 0.7)
        n_samples_test = n_samples - n_samples_train

        n_sub_train = n_samples_train // 3
        n_sub_test = n_samples_test // 2

        y_train = np.concatenate((np.zeros(n_sub_train), np.ones(n_sub_train), 2 * np.ones(n_samples_train - 2 * n_sub_train)))
        y_test = np.concatenate((np.zeros(n_sub_test), np.ones(n_samples_test - n_sub_test)))

        for idx, dist in enumerate(dists):
            gE.create_dataset(f'y{idx}', shape=y_train.shape, dtype=y_train.dtype, data=y_train)
            gO.create_dataset(f'y{idx}', shape=y_test.shape, dtype=y_test.dtype, data=y_test)
            X_train = np.concatenate((
                swiss_roll(n_sub_train, 0.2, 0.2),
                swiss_roll(n_sub_train, 0.5, 0.3),
                swiss_roll(n_samples_train - 2 * n_sub_train, 0.8, 0.2)
            ))
            offset_u = dist * 1
            offset_y = -dist * 0.3
            X_test = np.concatenate((
                swiss_roll(n_sub_test, 0.5, 0.3, offset_u, offset_y),
                swiss_roll(n_samples_test - n_sub_test, 0.2, 0.2, offset_u, offset_y),
            ))

            gE.create_dataset(f'X{idx}', shape=X_train.shape, dtype=X_train.dtype, data=X_train)
            gO.create_dataset(f'X{idx}', shape=X_test.shape, dtype=X_test.dtype, data=X_test)


def make_hybrid_dist(path, n_samples, width = 0.5, k = 0.7, u = 2.0):
    np.random.seed(0)
    generator = mtrand._rand

    t_max = 2.0 * np.pi

    def swiss_roll(n_samples, center1, center2, offset1=0, offset2=0):
        t = t_max * generator.normal(center1 + offset1, 0.2, size=n_samples)
        t = np.sort(t)
        y = width * generator.normal(center2 + offset2, 0.6, size=n_samples)
        x = k * t * np.cos(u * t)
        z = k * t * np.sin(u * t)
        X = np.vstack((x, y, z))
        X = X.T

        return X

    o = np.array([-1.0, -1.0, 0.0])
    n = np.array([0.0, 1.0, 0.01])
    # n /= np.linalg.norm(n)

    def plane_eq(x, y):
        return -(n[0] * (x - o[0]) + n[1] * (y - o[1])) / n[2] + o[2]

    # get point on plane
    p = np.array([0.0, 0.0, 0.0])
    p[2] = plane_eq(p[0], p[1])

    # get orthogonal basis
    p = p / np.linalg.norm(p)
    q = np.cross(n, p)
    q = q / np.linalg.norm(q)

    dists = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]

    with hf.File(path, 'w') as f:
        gE = f.create_group('E')
        gO = f.create_group('O')

        n_samples_train = int(n_samples * 0.7)
        n_samples_test = n_samples - n_samples_train
        n_sub_train = n_samples_train // 3 
        n_sub_test = n_samples_test // 3

        y_train = np.concatenate((
            np.zeros(n_sub_train), 
            np.ones(n_sub_train),
            2 * np.ones(n_samples_train- 2 * n_sub_train),
            ))
        y_test = np.concatenate((
            np.zeros(n_sub_test), 
            np.ones(n_sub_test),
            2 * np.ones(n_samples_test- 2 * n_sub_test),
            ))

        for idx, dist in enumerate(dists):
            gE.create_dataset(f'y{idx}', shape=y_train.shape, dtype=y_train.dtype, data=y_train)
            gO.create_dataset(f'y{idx}', shape=y_test.shape, dtype=y_test.dtype, data=y_test)

            # plane 2, roll 1
            sigma = 1.2
            train_set_p = generator.normal(0.0, sigma, size=n_sub_train)
            train_set_q = generator.normal(-0.5, sigma, size=n_sub_train)
            plane_train = train_set_p[:, np.newaxis] * p + train_set_q[:, np.newaxis] * q
            X_train = np.concatenate((
                swiss_roll(n_sub_train, 0.3, 0.3),
                swiss_roll(n_samples_train - 2 * n_sub_train, 0.5, 0.3),
                plane_train,
            ))

            offset_p = dist[0] * 1.0
            offset_q = dist[1] * 0.3
            offset_u = dist[2] * 1
            offset_y = -dist[2] * 0.3

            # plane 2, roll 1
            test_set_p = generator.normal(0.0 + offset_p, sigma, size=n_sub_test)
            test_set_q = generator.normal(-0.5 + offset_q, sigma, size=n_sub_test)
            plane_test = test_set_p[:, np.newaxis] * p + test_set_q[:, np.newaxis] * q
            X_test = np.concatenate((
                swiss_roll(n_sub_test, 0.3, 0.3, offset_u, offset_y),
                swiss_roll(n_samples_test - 2 * n_sub_test, 0.5, 0.3, offset_u, offset_y),
                plane_test,
            ))

            gE.create_dataset(f'X{idx}', shape=X_train.shape, dtype=X_train.dtype, data=X_train)
            gO.create_dataset(f'X{idx}', shape=X_test.shape, dtype=X_test.dtype, data=X_test)



if __name__ == '__main__':
    n_samples = 2000

    # make_plane_structure(f'datasets/synth/plane_structure_{n_samples}.h5', n_samples=n_samples)
    # with hf.File(f'datasets/synth/plane_structure_{n_samples}.h5', 'r') as f:
    #     fig = plt.figure()

    #     X  = np.array(f['E'][f'X0'])
    #     Xt = np.array(f['O'][f'X0'])

    #     yTrain = np.array(f['E'][f'y{0}'])
    #     yTest  = np.array(f['O'][f'y{0}'])

    #     plot = fig.add_subplot(111, projection='3d')
    #     plot.scatter(X[:, 0], X[:, 1], X[:, 2], s=2, c=yTrain,cmap='YlGn')
    #     plot.scatter(Xt[:, 0], Xt[:, 1], Xt[:, 2], s=2, c=yTest, cmap='Reds')

    #     plt.show()

    # make_swissroll_structure(f'datasets/synth/swissroll_structure_{n_samples}.h5', n_samples=n_samples)
    # with hf.File(f'datasets/synth/swissroll_structure_{n_samples}.h5', 'r') as f:
    #     fig = plt.figure()

    #     X  = np.array(f['E'][f'X0'])
    #     Xt = np.array(f['O'][f'X0'])

    #     yTrain = np.array(f['E'][f'y{0}'])
    #     yTest  = np.array(f['O'][f'y{0}'])

    #     plot = fig.add_subplot(111, projection='3d')
    #     plot.scatter(X[:, 0], X[:, 1], X[:, 2], s=2, c=yTrain,cmap='autumn')
    #     plot.scatter(Xt[:, 0], Xt[:, 1], Xt[:, 2], s=2, c=yTest, cmap='Oranges')

    #     plt.show()

    # make_hybrid_structure(f'datasets/synth/hybrid_structure_{n_samples}.h5', n_samples=n_samples)
    # with hf.File(f'datasets/synth/hybrid_structure_{n_samples}.h5', 'r') as f:
    #     fig = plt.figure()

    #     X  = np.array(f['E'][f'X0'])
    #     Xt = np.array(f['O'][f'X0'])

    #     yTrain = np.array(f['E'][f'y{0}'])
    #     yTest  = np.array(f['O'][f'y{0}'])

    #     plot = fig.add_subplot(111, projection='3d')
    #     plot.scatter(X[:, 0], X[:, 1], X[:, 2], s=2, c=yTrain,cmap='autumn')
    #     plot.scatter(Xt[:, 0], Xt[:, 1], Xt[:, 2], s=2, c=yTest, cmap='Oranges')

    #     plt.show()

    # make_plane_dist(f"datasets/synth/plane_dist_{n_samples}.h5", n_samples)
    # with hf.File(f'datasets/synth/plane_dist_{n_samples}.h5', 'r') as f:
    #     fig = plt.figure()

    #     plots = [fig.add_subplot(120 + i + 1, projection='3d') for i in range(2)]

    #     for oos_stage_idx, plot in enumerate(plots):
    #         X  = np.array(f['E'][f'X{oos_stage_idx}'])
    #         Xt = np.array(f['O'][f'X{oos_stage_idx}'])

    #         yTrain = np.array(f['E'][f'y{oos_stage_idx}'])
    #         yTest  = np.array(f['O'][f'y{oos_stage_idx}'])

    #         plot.scatter(X[:, 0], X[:, 1], X[:, 2], s=2, c=yTrain,cmap='autumn')
    #         plot.scatter(Xt[:, 0], Xt[:, 1], Xt[:, 2], s=2, c=yTest, cmap='Oranges')

    #     plt.show()

    # make_swissroll_dist(f'datasets/synth/swissroll_dist_{n_samples}.h5', n_samples=n_samples, k=0.8, u=1.5)
    # with hf.File(f'datasets/synth/swissroll_dist_{n_samples}.h5', 'r') as f:
    #     fig = plt.figure()

    #     plots = [fig.add_subplot(120 + i + 1, projection='3d') for i in range(2)]

    #     for oos_stage_idx, plot in enumerate(plots):
    #         X  = np.array(f['E'][f'X{oos_stage_idx}'])
    #         Xt = np.array(f['O'][f'X{oos_stage_idx}'])

    #         yTrain = np.array(f['E'][f'y{oos_stage_idx}'])
    #         yTest  = np.array(f['O'][f'y{oos_stage_idx}'])

    #         plot.scatter(X[:, 0], X[:, 1], X[:, 2], s=2, c=yTrain,cmap='autumn')
    #         plot.scatter(Xt[:, 0], Xt[:, 1], Xt[:, 2], s=2, c=yTest, cmap='Oranges')

    #     plt.show()

    make_hybrid_dist(f'datasets/synth/hybrid_dist_{n_samples}.h5', n_samples=n_samples, k=0.8, u=1.5)
    with hf.File(f'datasets/synth/hybrid_dist_{n_samples}.h5', 'r') as f:
        fig = plt.figure()

        plots = [fig.add_subplot(120 + i + 1, projection='3d') for i in range(2)]

        for oos_stage_idx, plot in enumerate(plots):
            X  = np.array(f['E'][f'X{oos_stage_idx}'])
            Xt = np.array(f['O'][f'X{oos_stage_idx}'])

            yTrain = np.array(f['E'][f'y{oos_stage_idx}'])
            yTest  = np.array(f['O'][f'y{oos_stage_idx}'])

            plot.scatter(X[:, 0], X[:, 1], X[:, 2], s=2, c=yTrain,cmap='autumn')
            plot.scatter(Xt[:, 0], Xt[:, 1], Xt[:, 2], s=2, c=yTest, cmap='Oranges')

        plt.show()

    # make_plane_prop(f"datasets/synth/plane_prop_{n_samples}.h5", n_samples=n_samples)
    # with hf.File(f'datasets/synth/plane_prop_{n_samples}.h5', 'r') as f:
    #     fig = plt.figure()

    #     plots = [fig.add_subplot(140 + i + 1, projection='3d') for i in range(4)]

    #     for oos_stage_idx, plot in enumerate(plots):
    #         X  = np.array(f['E'][f'X{oos_stage_idx}'])
    #         Xt = np.array(f['O'][f'X{oos_stage_idx}'])

    #         yTrain = np.array(f['E'][f'y{oos_stage_idx}'])
    #         yTest  = np.array(f['O'][f'y{oos_stage_idx}'])

    #         plot.scatter(X[:, 0], X[:, 1], X[:, 2], s=2, c=yTrain,cmap='autumn')
    #         plot.scatter(Xt[:, 0], Xt[:, 1], Xt[:, 2], s=2, c=yTest, cmap='Oranges')

    #     plt.show()

    # make_swissroll_prop(f"datasets/synth/swissroll_prop_{n_samples}.h5", n_samples=n_samples)
    # with hf.File(f'datasets/synth/swissroll_prop_{n_samples}.h5', 'r') as f:
    #     fig = plt.figure()

    #     plots = [fig.add_subplot(140 + i + 1, projection='3d') for i in range(4)]

    #     for oos_stage_idx, plot in enumerate(plots):
    #         X  = np.array(f['E'][f'X{oos_stage_idx}'])
    #         Xt = np.array(f['O'][f'X{oos_stage_idx}'])

    #         yTrain = np.array(f['E'][f'y{oos_stage_idx}'])
    #         yTest  = np.array(f['O'][f'y{oos_stage_idx}'])

    #         plot.scatter(X[:, 0], X[:, 1], X[:, 2], s=2, c=yTrain,cmap='autumn')
    #         plot.scatter(Xt[:, 0], Xt[:, 1], Xt[:, 2], s=2, c=yTest, cmap='Oranges')

    #     plt.show()

    # make_hybrid_prop(f'datasets/synth/hybrid_prop_{n_samples}.h5', n_samples=n_samples)
    # with hf.File(f'datasets/synth/hybrid_prop_{n_samples}.h5', 'r') as f:
    #     fig = plt.figure()

    #     plots = [fig.add_subplot(140 + i + 1, projection='3d') for i in range(4)]

    #     for oos_stage_idx, plot in enumerate(plots):
    #         yTrain = np.array(f['E'][f'y{oos_stage_idx}'])
    #         yTest  = np.array(f['O'][f'y{oos_stage_idx}'])

    #         X  = np.array(f['E'][f'X{oos_stage_idx}'])
    #         Xt = np.array(f['O'][f'X{oos_stage_idx}'])

    #         plot.scatter(X[:, 0], X[:, 1], X[:, 2], s=2, c=yTrain,cmap='autumn')
    #         plot.scatter(Xt[:, 0], Xt[:, 1], Xt[:, 2], s=2, c=yTest, cmap='Oranges')

    #     plt.show()
