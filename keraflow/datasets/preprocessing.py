import numpy as np


def pad_sequences(sentences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    '''Taken from Keras.
    Pads each sentece to the length of the longest sentence.
    If maxlen is provided, any sequence longer than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or the end of the sequence.
    Supports post-padding and pre-padding (default).

    @param sentences: list of lists where each element is a sequence
    @param maxlen: int, maximum length
    @param dtype: type to cast the resulting sequence.
    @param padding: 'pre' or 'post', pad either before or after each sequence.
    @param truncating: 'pre' or 'post', remove values from sentences larger than maxlen either in the beginning or in the end of the sequence
    @param value: float, value to pad the sentences to the desired value.
    @return numpy array. Shape: `(number_of_sequences, maxlen)`
    '''
    lengths = [len(s) for s in sentences]

    nb_samples = len(sentences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sentences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sentences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x
