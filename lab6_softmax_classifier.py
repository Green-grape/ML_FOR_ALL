import tensorflow as tf
import numpy as np

x_data = np.array([[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5], [
                  1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]], dtype=np.float32)
y_data = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [
                  0, 1, 0], [1, 0, 0], [1, 0, 0]], dtype=np.float32)

nb_classes = 3
W = tf.Variable(tf.random.normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random.normal([nb_classes]), name='bias')


def hypothesis(X):
    return tf.nn.softmax(tf.matmul(X, W)+b)


learning_rate = 0.01
n_epochs = 1000
for i in range(n_epochs):
    with tf.GradientTape() as tape:
        cost = tf.reduce_mean(-tf.reduce_sum(y_data *
                              tf.math.log(hypothesis(x_data)), axis=1))
    W_grad, b_grad = tape.gradient(cost, [W, b])
    W.assign_sub(learning_rate*W_grad)
    b.assign_sub(learning_rate*b_grad)


def accuracy(y, y_hat):
    predicted = tf.cast(hypothesis(y_hat) > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(predicted, y), dtype=tf.float32))
    return accuracy


print('Accuracy:', accuracy(y_data, x_data).numpy())
