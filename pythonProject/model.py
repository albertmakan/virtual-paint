from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten


def create_model():
    cnn_model = Sequential()
    cnn_model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(120, 120, 3)))
    cnn_model.add(MaxPooling2D((2, 2)))

    cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D((2, 2)))

    cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Flatten())

    cnn_model.add(Dense(128, activation='relu'))
    cnn_model.add(Dense(6, activation='softmax'))
    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    cnn_model.summary()
    return cnn_model
