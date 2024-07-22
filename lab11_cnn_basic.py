import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# 하나의 이미지, 3x3x1 크기의 이미지
image = np.array(
    [[[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]], dtype=np.float32)

# 2x2 크기의 필터
# 3개의 필터를 사용
fil = tf.constant([[[[1., 10., -1.]], [[1., 10., -1.]]],
                  [[[1., 10., -1.]], [[1., 10., -1.]]]], dtype=np.float32)

# padding: SAME (원본 이미지와 같은 크기로 출력)
# padding: VALID (필터 크기만큼 줄어든 이미지로 출력)
conv2d = tf.nn.conv2d(image, fil, strides=[1, 1, 1, 1], padding='SAME')
conv2d_img = conv2d.numpy()

# pooling
img = tf.constant([[[[4], [3]], [[2], [1]]]], dtype=np.float32)
# ksize: pooling 크기, strides: 이동 크기
pool = tf.nn.max_pool(img, ksize=[1, 2, 2, 1], strides=[
                      1, 1, 1, 1], padding='SAME')
print(pool)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

img = x_train[0].reshape(28, 28)
plt.imshow(img, cmap='gray')
plt.show()

# MNIST convolution layer
img = img.reshape(-1, 28, 28, 1)  # 28x28x1
W1 = tf.Variable(tf.random.normal([3, 3, 1, 5], stddev=0.01))  # 3x3x1, 5개
conv2d = tf.nn.conv2d(img, W1, strides=[1, 2, 2, 1], padding='SAME')  # 14x14x5
conv2d_img = np.swapaxes(conv2d, 0, 3)
for i, one_img in enumerate(conv2d_img):
    plt.subplot(1, 5, i+1)
    plt.imshow(one_img.reshape(14, 14), cmap='gray')
plt.show()

# MNIST pooling layer
pool = tf.nn.max_pool(conv2d, ksize=[1, 2, 2, 1], strides=[
                      1, 2, 2, 1], padding='SAME')  # 7x7x5
pool_img = np.swapaxes(pool, 0, 3)
for i, one_img in enumerate(pool_img):
    plt.subplot(1, 5, i+1)
    plt.imshow(one_img.reshape(7, 7), cmap='gray')
plt.show()
