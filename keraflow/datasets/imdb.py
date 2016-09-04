import numpy as np

from six.moves import cPickle, zip

from .common import download


def load_data(nb_words=None, skip_top=0, maxlen=None, start_char=1, oov_char=2, index_from=3):
    '''Taken from Keras.
    @param nb_words: max number of words to include. Words are ranked by how often they occur (in the training set) and only the most frequent words are kept
    @param skip_top: skip the top N most frequently occuring words (which may not be informative).
    @param maxlen: truncate sequences after this length.
    @param start_char: The start of a sequence will be marked with this character.  Set to 1 because 0 is usually the padding character.
    @param oov_char: words that were cut out because of the `nb_words` or `skip_top` limit will be replaced with this character.
    @param index_from: index actual words with this index and higher.

    Note that the 'out of vocabulary' character is only used for words that were present in the training set but are not included because they're not making the `nb_words` cut here.  Words that were not seen in the trining set but are in the test set have simply been skipped.
    '''
    path = download("https://raw.githubusercontent.com/ipod825/keraflow_dataset/master/datasets/imdb_full.pkl")
    with open(path) as f:
        (x_train, labels_train), (x_test, labels_test) = cPickle.load(f)

    X = x_train + x_test
    labels = labels_train + labels_test

    if start_char is not None:
        X = [[start_char] + [w + index_from for w in x] for x in X]
    elif index_from:
        X = [[w + index_from for w in x] for x in X]

    if maxlen:
        new_X = []
        new_labels = []
        for x, y in zip(X, labels):
            if len(x) < maxlen:
                new_X.append(x)
                new_labels.append(y)
        X = new_X
        labels = new_labels
    if not X:
        raise Exception('After filtering for sequences shorter than maxlen={}, no sequence was kept. Increase maxlen.'.format(maxlen))
    if not nb_words:
        nb_words = max([max(x) for x in X])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters: 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        X = [[oov_char if (w >= nb_words or w < skip_top) else w for w in x] for x in X]
    else:
        nX = []
        for x in X:
            nx = []
            for w in x:
                if (w >= nb_words or w < skip_top):
                    nx.append(w)
            nX.append(nx)
        X = nX

    X_train = np.array(X[:len(x_train)])
    y_train = np.array(labels[:len(x_train)])

    X_test = np.array(X[len(x_train):])
    y_test = np.array(labels[len(x_train):])

    return (X_train, y_train), (X_test, y_test)
