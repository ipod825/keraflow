# -*- coding: utf-8 -*-
# from six.moves import cPickle
import cPickle
import gzip

from .common import download


def load_data():
    path = download("https://raw.githubusercontent.com/ipod825/keraflow_dataset/master/datasets/mnist.pkl.gz")

    f = gzip.open(path, 'rb')
    data = cPickle.load(f)
    f.close()
    return data  # (X_train, y_train), (X_test, y_test)
