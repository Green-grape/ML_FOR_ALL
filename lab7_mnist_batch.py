import tensorflow as tf
import numpy as np
import os
from datetime import datetime

minst = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = minst.load_data()

#  Normalization
x_train, x_test = x_train / 255.0, x_test / 255.0

print(y_train, y_test)

# one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # 28*28=784을 1차원으로 변환
model.add(tf.keras.layers.Dense(units=10, input_dim=28*28,
          activation='softmax'))  # 10개의 출력을 가지는 softmax layer

batch_size = 100
n_epochs = 5
optimizer = tf.keras.optimizers.SGD(lr=1e-3)
loss_fn = tf.keras.losses.CategoricalCrossentropy()


for i in range(n_epochs):
    print("\nStart of epoch %d" % i,)
    for j in range(0, len(x_train), batch_size):
        with tf.GradientTape() as tape:
            logits = model(x_train[j:j+batch_size], training=True)
            loss_value = loss_fn(y_train[j:j+batch_size], logits)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print("Epoch: ", i, "Loss: ", loss_value)


print("Accuracy: ", model.evaluate(x_test, y_test)[1])
