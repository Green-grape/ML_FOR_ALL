import tensorflow as tf
import numpy as np

num_classes = 5
input_dim = 5  # one-hot size
sequence_length = 6  # 'hihell'
learning_rate = 0.1

idx2char = ['h', 'i', 'e', 'l', 'o']
x_data = [[0, 1, 0, 2, 3, 3]]  # hihell
x_one_hot = np.array([[[1, 0, 0, 0, 0],  # h 0
                       [0, 1, 0, 0, 0],  # i 1
                       [1, 0, 0, 0, 0],  # h 0
                       [0, 0, 1, 0, 0],  # e 2
                       [0, 0, 0, 1, 0],  # l 3
                       [0, 0, 0, 1, 0]]],  # l 3
                     dtype=np.float32)
y_one_hot = tf.keras.utils.to_categorical(
    [[1, 0, 2, 3, 3, 4]], num_classes=num_classes)
tf.model = tf.keras.Sequential()
cell = tf.keras.layers.LSTMCell(
    units=num_classes, input_shape=(sequence_length, input_dim))
tf.model.add(tf.keras.layers.RNN(cell, return_sequences=True))
tf.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(
    units=num_classes, activation='softmax')))  # 각 timestamp마다 cost를 계산하기 위해 사용
tf.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                 metrics=['accuracy'])

history = tf.model.fit(x_one_hot, y_one_hot, epochs=2000)
tf.model.summary()
predictions = tf.model.predict(x_one_hot)
for i, prediction in enumerate(predictions):
    print(prediction)
    result_str = [idx2char[c] for c in np.argmax(prediction, axis=1)]
    print("\tPrediction str: ", ''.join(result_str))
