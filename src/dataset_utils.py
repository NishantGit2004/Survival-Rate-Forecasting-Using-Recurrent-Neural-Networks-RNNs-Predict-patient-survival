import numpy as np
import tensorflow as tf

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    X = d['X']
    y = d['y']
    lengths = d['lengths']
    ids = d['ids']
    return X, y, lengths, ids

def to_tf_dataset(X, y, batch_size=32, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
