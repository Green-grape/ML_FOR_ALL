# Lab 4 Multi-variable linear regression
import tensorflow as tf
import numpy as np

file_paths = ['./data/data-01-test-score.csv',
              './data/data-01-test-score_copy.csv', './data/data-01-test-score_copy2.csv']
num_epochs = 100


def tf_parse_filename(filename):

    def parse_filename(filename_batch):
        data = []
        labels = []
        for filename in filename_batch:
            # Read data
            filename_str = filename.numpy().decode()
            # Read .csv file
            data_point = np.loadtxt(filename_str, delimiter=',')
            print('data_point:', data_point)
            data.append(data_point[:-1])
            labels.append(data_point[-1])
        print(data, labels)
        return np.stack(data), np.stack(labels)
    x, y = tf.py_function(parse_filename, [filename], [
                          tf.int32, tf.int32])
    print('x, y:', x, y)
    return x, y


train_ds = tf.data.Dataset.from_tensor_slices(file_paths)
train_ds = train_ds.batch(32, drop_remainder=True)
train_ds = train_ds.map(tf_parse_filename, num_parallel_calls=1)
train_ds = train_ds.prefetch(buffer_size=32)

# Train on epochs
for i in range(num_epochs):
    # Train on batches
    for x_train, y_train in train_ds:
        print(x_train, y_train)
        # train_on_batch(x_train, y_train)

print('Training done!')
exit()

'''
[[ 73.  80.  75.]
 [ 93.  88.  93.]
 ...
 [ 76.  83.  71.]
 [ 96.  93.  95.]] 
x_data shape: (25, 3)
[[152.]
 [185.]
 ...
 [149.]
 [192.]] 
y_data shape: (25, 1)
'''
tf.model = tf.keras.Sequential()
# activation function doesn't have to be added as a separate layer. Add it as an argument of Dense() layer
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=3, activation='linear'))
# tf.model.add(tf.keras.layers.Activation('linear'))
tf.model.summary()

tf.model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(lr=1e-5))
history = tf.model.fit(x_data, y_data, epochs=2000)

# Ask my score
print("Your score will be ", tf.model.predict([[100, 70, 101]]))
print("Other scores will be ", tf.model.predict(
    [[60, 70, 110], [90, 100, 80]]))
