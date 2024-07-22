import tensorflow as tf
import numpy as np
import os
from datetime import datetime

minst = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = minst.load_data()

#  Normalization
x_train, x_test = x_train / 255.0, x_test / 255.0

# one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # 28*28=784을 1차원으로 변환
model.add(tf.keras.layers.Dense(units=256, input_dim=28*28,
          activation='relu'))  # 10개의 출력을 가지는 relu layer
model.add(tf.keras.layers.Dense(units=256, input_dim=256,
          activation='relu'))  # 10개의 출력을 가지는 relu layer
model.add(tf.keras.layers.Dense(units=10, input_dim=256, activation='softmax'))

log_dir = os.path.join(
    ".", "logs", "mnist", "fit", datetime.now().strftime("%Y%m%d-%H%M%S"))

batch_size = 100
n_epochs = 15
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
loss_fn = tf.keras.losses.CategoricalCrossentropy()


model.compile(loss=loss_fn,
              optimizer=optimizer, metrics=['accuracy'])

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)

model.fit(x_train, y_train, epochs=n_epochs,
          batch_size=batch_size, callbacks=[tensorboard_callback])

print("Accuracy: ", model.evaluate(x_test, y_test)[1])
