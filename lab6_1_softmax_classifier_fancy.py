import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# Predicting animal type based on various features
xy = np.loadtxt('./data/data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42)

print(x_data.shape, y_data.shape)

nb_classes = 7  # 0 ~ 6
n_features = x_data.shape[1]

y_one_hot = tf.keras.utils.to_categorical(y_train, nb_classes)
print("one_hot:", y_one_hot.shape)

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=nb_classes,
             input_dim=n_features, activation='softmax'))
tf.model.compile(loss='categorical_crossentropy',
                 optimizer=tf.keras.optimizers.SGD(lr=0.1), metrics=['accuracy'])
tf.model.summary()

history = tf.model.fit(x_train, y_one_hot, epochs=1000)

# testing
y_one_hot = tf.keras.utils.to_categorical(y_test, nb_classes)
print("Testing Accuracy: ", tf.model.evaluate(x_test, y_one_hot)[1])
