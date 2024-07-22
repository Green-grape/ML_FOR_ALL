import tensorflow as tf
import numpy as np

x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

tf.model = tf.keras.Sequential()

tf.model.add(tf.keras.layers.Dense(units=1, input_dim=3))
tf.model.add(tf.keras.layers.Activation('linear'))


# adam -> optimizaer 이후 추가 공부 필요
tf.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=1e-3))

tf.model.summary()
history = tf.model.fit(x_data, y_data, epochs=10000)
y_predict = tf.model.predict(np.array([[72., 93., 90.]]))
print(y_predict)
