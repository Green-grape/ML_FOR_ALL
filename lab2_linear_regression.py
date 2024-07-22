import numpy as np
import tensorflow as tf

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

tf.model = tf.keras.Sequential()  # layer를 쌓아갈 모델
# units=1은 뉴런수 input_dim=1은 입력이 1차원이라는 뜻
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=1))

# SGD는 경사하강법(standard gradient descent), lr은 learning rate
sgd = tf.keras.optimizers.SGD(lr=0.1)
# loss: MSE(Mean Squared Error), optimizer: sgd
tf.model.compile(loss='mse', optimizer=sgd)

tf.model.summary()

tf.model.fit(x_train, y_train, epochs=200)
y_predict = tf.model.predict(np.array([5, 4]))
print(y_predict)
