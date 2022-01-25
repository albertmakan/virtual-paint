from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, BatchNormalization, Dropout


def create_model_small(gray=False):
    cnn_model = Sequential()
    cnn_model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(120, 120, 1 if gray else 3)))
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


def create_model_large(gray=False):
    cnn_model = Sequential()
    cnn_model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(120, 120, 1 if gray else 3)))
    cnn_model.add(BatchNormalization())

    cnn_model.add(Conv2D(16, (3, 3), activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D((2, 2)))

    cnn_model.add(Conv2D(32, (2, 2), activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D((2, 2)))

    cnn_model.add(Conv2D(64, (2, 2), activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D((2, 2)))

    cnn_model.add(Conv2D(128, (2, 2), activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D((2, 2)))

    cnn_model.add(Flatten())

    cnn_model.add(Dense(512, activation='relu'))
    cnn_model.add(Dropout(0.5))

    cnn_model.add(Dense(256, activation='relu'))
    cnn_model.add(Dropout(0.5))

    cnn_model.add(Dense(6, activation='softmax'))
    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    cnn_model.summary()

    return cnn_model
