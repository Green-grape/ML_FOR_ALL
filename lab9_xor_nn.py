import tensorflow as tf
import numpy as np
import os
from datetime import datetime

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=10, input_dim=2))
tf.model.add(tf.keras.layers.Activation('sigmoid'))
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=10))
tf.model.add(tf.keras.layers.Activation('sigmoid'))
tf.model.compile(loss='binary_crossentropy',
                 optimizer=tf.optimizers.SGD(lr=0.01), metrics=['accuracy'])
tf.model.summary()

# add tensorboard
log_dir = os.path.join(
    ".", "logs", "fit", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)

history = tf.model.fit(x_data, y_data, epochs=30000,
                       verbose=0, callbacks=[tensorboard_callback])
predictions = tf.model.predict(x_data)
print('score:', tf.model.evaluate(x_data, y_data))
