import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

tf.random.set_seed(777)

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

training_epochs = 30
batch_size = 100


# Model definition using Keras API in TensorFlow 2.x
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, input_dim=784, activation='relu', input_shape=(
        784,), kernel_initializer='glorot_normal', bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None), kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu', bias_initializer=tf.keras.initializers.RandomNormal(
        mean=0.0, stddev=1.0, seed=None), kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax', kernel_initializer='glorot_normal',
                          bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None), kernel_regularizer=tf.keras.regularizers.l2(0.01))
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=training_epochs,
          batch_size=batch_size, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Accuracy:', test_acc)
