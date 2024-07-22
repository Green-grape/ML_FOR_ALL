import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

xy = np.loadtxt('./data/data-02-stock_daily.csv', delimiter=',')

timesteps = seq_length = 7
data_dim = 5
output_dim = 1
xy = xy[::-1]  # reverse order (chronically ordered)
xy = MinMaxScaler().fit_transform(xy)
x = xy
y = xy[:, [-1]]  # Close as label

train_size = int(len(y)*0.7)
x_train, x_test = x[0:train_size], x[train_size:]
y_train, y_test = y[0:train_size], y[train_size:]


def build_dataset(x, y, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(y) - seq_length):
        _x = x[i:i + seq_length]
        _y = y[i + seq_length]  # Next close price
        print(_x, "->", _y)
        dataX.append(_x)
        dataY.append(_y)
    return np.array(dataX), np.array(dataY)


x_train, y_train = build_dataset(x_train, y_train, seq_length)
x_test, y_test = build_dataset(x_test, y_test, seq_length)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.LSTM(units=1, input_shape=(seq_length, data_dim)))
tf.model.add(tf.keras.layers.Dense(units=output_dim, activation='tanh'))
tf.model.compile(loss='mean_squared_error',
                 optimizer=tf.keras.optimizers.Adam(lr=0.01))
tf.model.fit(x_train, y_train, epochs=500)

y_predict = tf.model.predict(x_test)

plt.plot(y_test, label='actual')
plt.plot(y_predict, label='predict')
plt.xlabel("Time Period")
plt.ylabel("Stock Price")
plt.legend()
plt.title("Stock Price Prediction")
plt.show()
