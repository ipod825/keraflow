'''
Taken from Keras.
This example demonstrates the use of Convolution1D for text classification.
'''

import keraflow
from keraflow.models import Sequential
from keraflow.layers import Input, Dense, Dropout, Activation, Flatten
from keraflow.layers import Embedding
from keraflow.layers import Convolution1D, Pooling1D
from keraflow.datasets import imdb
from keraflow.datasets.preprocessing import pad_sequences

keraflow.seed(1337)  # for reproducibility


# set parameters:
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
nb_kernel = 250
kernel_row = 3
hidden_dims = 250
nb_epoch = 2

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

# import numpy as np
# X_train = np.concatenate((X_train[:100],X_train[-100:]), axis=0)
# y_train = np.concatenate((y_train[:100],y_train[-100:]), axis=0)
# X_test = np.concatenate((X_test[:10],X_test[-10:]), axis=0)
# y_test = np.concatenate((y_test[:10],y_test[-10:]), axis=0)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()

model.add(Input(maxlen))

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features, embedding_dims, dropout=0.2))

# we add a Convolution1D, which will learn nb_kernel
# word group filters of size kernel_row:
model.add(Convolution1D(nb_kernel=nb_kernel,
                        kernel_row=kernel_row,
                        pad='valid',
                        activation='relu',
                        stride=1))
# we use max pooling:
model.add(Pooling1D('max', pool_length=maxlen-2))

# We flatten the output of the conv layer,
# so that we can add a vanilla dense layer:
model.add(Flatten())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))
