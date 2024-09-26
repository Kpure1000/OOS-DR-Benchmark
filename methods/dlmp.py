from keras import datasets as kdatasets
from keras.callbacks import EarlyStopping
from keras.initializers import Constant
from keras.layers import Dense, Dropout
from keras.models import Sequential
import tensorflow as tf


def train_model(X, X_2d, loss='mean_squared_error', epochs=None, batch_size=32, lr=1e-3, verbose=False):
    callbacks = []

    stop = EarlyStopping(verbose=1, min_delta=0.00001, mode='min', patience=10, restore_best_weights=True)
    callbacks.append(stop)

    m = Sequential()
    m.add(Dense(256, activation='relu',
                kernel_initializer='he_uniform',
                bias_initializer=Constant(0.0001),
                input_shape=(X.shape[1],)))
    m.add(Dense(512, activation='relu',
                kernel_initializer='he_uniform',
                bias_initializer=Constant(0.0001)))
    m.add(Dense(256, activation='relu',
                kernel_initializer='he_uniform',
                bias_initializer=Constant(0.0001)))
    m.add(Dense(2, activation='sigmoid',
                kernel_initializer='he_uniform',
                bias_initializer=Constant(0.0001)))
    m.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

    hist = m.fit(X,
                 X_2d,
                 batch_size=batch_size,
                 epochs=200 if not epochs else epochs,
                 verbose=1 if verbose else 0,
                 validation_split=0.05,
                 callbacks=callbacks,
                 use_multiprocessing=True,
                 workers=4)

    return m, hist
