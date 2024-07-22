import tensorflow as tf
import numpy as np

hidden_size = 2
batch_size = 3

h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]
# shape: (1, 5, 4), 1:batch size, 5: sequence_length, 4: input_dim
x_data = np.array([[h, e, l, l, o], [e, o, l, l, l],
                  [l, l, e, e, l]], dtype=np.float32)
print(x_data.shape)

tf.model = tf.keras.Sequential()
cell = tf.keras.layers.LSTMCell(units=hidden_size, input_shape=(5, 4))
tf.model.add(tf.keras.layers.RNN(
    cell, return_sequences=True))
tf.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.1))

# (1, 5, 2) 1:batch size 5: sequence_length, 2: hidden_size
print(tf.model.predict(x_data).shape)
