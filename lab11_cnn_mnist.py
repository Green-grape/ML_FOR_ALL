import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


def create_model():
    model = tf.keras.models.Sequential()

    # L1 Conv layer
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(
        3, 3), input_shape=(28, 28, 1), activation='relu', padding='SAME'))  # 28x28x1 -> 28x28x32
    model.add(tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2), padding='SAME'))  # 28x28x32 -> 14x14x32

    # L2 Conv layer
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(
        3, 3), activation='relu', padding='SAME'))  # 14x14x32 -> 14x14x64
    model.add(tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2), padding='SAME'))  # 14x14x64 -> 7x7x64

    # L3 Fully Connected (FC) layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(
        units=10, activation='softmax', kernel_initializer='glorot_normal'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])

    return model


models = [create_model() for _ in range(3)]
for model in models:
    model.fit(x_train, y_train, batch_size=100, epochs=15)

# ensemble prediction
y_predicted = np.array([model.predict(x_test) for model in models])
y_predicted = np.mean(y_predicted, axis=0)  # 3개의 모델의 예측값을 평균내어 사용
y_predicted = np.argmax(y_predicted, axis=1)

# calculate the accuracy
accuracy = np.mean(y_predicted == np.argmax(y_test, axis=1))
print('Accuracy:', accuracy)
