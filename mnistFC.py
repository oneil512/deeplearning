from keras.datasets import mnist
import keras.models as models
import keras.layers as layers


(x_train, y_train), (x_test, y_test) = mnist.load_data()

print('x_train', x_train.shape)
print('y_train', y_train.shape)

print('x_test', x_test.shape)
print('y_test', y_test.shape)

print()
print('reshape')
x_train = x_train.reshape((60000, 28 * 28))
x_train = x_train.astype('float32') / 255

x_test = x_test.reshape((10000, 28 * 28))
x_test = x_test.astype('float32') / 255

print('x_train', x_train.shape)
print('y_train', y_train.shape)

print('x_test', x_test.shape)
print('y_test', y_test.shape)
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

network.fit(x_train, y_train, epochs=5, batch_size=128)
