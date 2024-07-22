import tensorflow as tf
import numpy as np


learning_rate = 0.001
n_epochs = 1000

x_data = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [
                  5, 3], [6, 2]], dtype=np.float32)
y_data = np.array([[0], [0], [0], [1], [1], [1]], dtype=np.float32)

W = tf.Variable(tf.random.normal([2, 1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')


def hypothesis(X):
    return tf.sigmoid(tf.matmul(X, W) + b)


for i in range(n_epochs):
    with tf.GradientTape() as tape:
        cost = -tf.reduce_mean(y_data*tf.math.log(hypothesis(x_data)) +
                               (1-y_data)*tf.math.log(1-hypothesis(x_data)))
    W_grad, b_grad = tape.gradient(cost, [W, b])
    W.assign_sub(learning_rate*W_grad)
    b.assign_sub(learning_rate*b_grad)

text_x = np.array([[7, 6]], dtype=np.float32)
print(hypothesis(text_x).numpy()[0][0] > 0.5)
