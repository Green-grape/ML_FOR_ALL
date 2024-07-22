import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
tf.print(hello)


@tf.function
def adder(a, b):
    return a+b


a = tf.constant([1, 3])
b = tf.constant([2, 4])
print(a, b)
print(adder(a, b))
