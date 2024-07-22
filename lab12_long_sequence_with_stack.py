import tensorflow as tf
import numpy as np

sample = "if you want you to build a ship, don't drum up people together to collect wood and don't assign them tasks and work, but rather teach them to long for the endless immensity of the sea."
char_set = list(set(sample))  # index -> char
char_dic = {c: i for i, c in enumerate(char_set)}  # char -> index

# hyper parameters
dic_size = len(char_dic)
rnn_hidden_size = len(char_dic)
num_classes = len(char_dic)
max_sequence_length = 10  # Any arbitrary number
learning_rate = 0.1

x_data = []
y_data = []
for i in range(0, len(sample) - max_sequence_length):
    x_str = sample[i:i + max_sequence_length]
    y_str = sample[i + 1: i + max_sequence_length + 1]
    x = [char_dic[c] for c in x_str]
    y = [char_dic[c] for c in y_str]
    x_data.append(x)
    y_data.append(y)

# pad
x_data = tf.keras.preprocessing.sequence.pad_sequences(
    x_data, maxlen=max_sequence_length, padding='post', value=0)
y_data = tf.keras.preprocessing.sequence.pad_sequences(
    y_data, maxlen=max_sequence_length, padding='post', value=0)

x_data = tf.keras.utils.to_categorical(
    np.array(x_data), num_classes=num_classes)
y_data = tf.keras.utils.to_categorical(
    np.array(y_data), num_classes=num_classes)
batch_size = len(x_data)

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Masking(
    mask_value=0., input_shape=(None, dic_size)))
tf.model.add(tf.keras.layers.GRU(
    units=rnn_hidden_size, return_sequences=True))
tf.model.add(tf.keras.layers.GRU(
    units=rnn_hidden_size, return_sequences=True))
tf.model.add(tf.keras.layers.TimeDistributed(
    tf.keras.layers.Dense(units=num_classes, activation='softmax')))
tf.model.compile(loss='categorical_crossentropy',
                 optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
history = tf.model.fit(x_data, y_data, epochs=500)
predictions = tf.model.predict(x_data)
for i, prediction in enumerate(predictions):
    index = np.argmax(prediction, axis=1)
    if i == 0:
        print(''.join([char_set[t] for t in index]), end='')
    else:
        print(char_set[index[-1]], end='')
