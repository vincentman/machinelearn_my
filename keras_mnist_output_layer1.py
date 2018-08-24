import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt

np.random.seed(0)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dropout, Flatten

from keras.utils import np_utils

input_shape = (28, 28, 1)
batch_size = 128
hidden_neurons = 200
classes = 10
epochs = 1

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train[:1000]
Y_train = Y_train[:1000]

X_train = X_train.reshape(len(X_train), 28, 28, 1)
X_test = X_test.reshape(len(X_test), 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(Y_train, classes)
Y_test = np_utils.to_categorical(Y_test, classes)

model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(hidden_neurons))
model.add(Activation('relu'))
model.add(Dense(classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adagrad')

model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=1)

input_data = model.input
output = model.layers[0].output
functor = K.function([input_data] + [K.learning_phase()], [output])
data = X_train[0][np.newaxis, ...]
layer_out = functor([data, 1.])

plt.imshow(layer_out[0][0][:, :, 0])

print('end')
