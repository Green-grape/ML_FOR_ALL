import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

xy = np.loadtxt('./data/data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
print(x_data.shape, y_data.shape)
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=0)

W = tf.Variable(tf.random.normal([x_data.shape[1], 1]), name='weight')
b = tf.Variable(tf.random.normal([y_data.shape[1], 1]), name='bias')


def hypothesis(X):
    return tf.sigmoid(tf.matmul(X, W) + b)


learning_rate = 0.01
n_epochs = 1000
for i in range(n_epochs):
    with tf.GradientTape() as tape:
        cost = -tf.reduce_mean(y_train*tf.math.log(hypothesis(x_train)) +
                               (1-y_train)*tf.math.log(1-hypothesis(x_train)))
    W_grad, b_grad = tape.gradient(cost, [W, b])
    W.assign_sub(learning_rate*W_grad)
    b.assign_sub(learning_rate*b_grad)


def accuracy(y, y_hat):
    predicted = tf.cast(hypothesis(y_hat) > 0.5,
                        dtype=tf.float32)  # 0.5보다 크면 1, 작으면 0
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(predicted, y), dtype=tf.float32))  # 평균
    return accuracy


print('Accuracy:', accuracy(y_test, x_test).numpy())
