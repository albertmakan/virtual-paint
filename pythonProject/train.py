import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import load_model

from model import create_model


def read_data():
    labels = []
    images = []
    for i, c in enumerate(["choose", "draw", "erase", "move", "nothing", "select"]):
        folder_path = "../dataset/" + c
        for file in os.listdir(folder_path):
            labels.append(i)
            image = cv2.imread(f"{folder_path}/{file}")
            images.append(image)
    images = np.array(images, dtype="uint8")
    labels = np.array(labels)
    return images, labels


def train(experiment: str, save_path: str = "models/", load_path: str = None, batch_size=32, epochs=10):
    images, labels = read_data()
    images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.15)
    model = create_model() if load_path is None else load_model(load_path)
    history = model.fit(images_train, labels_train, epochs=epochs, batch_size=batch_size, verbose=2)
    model.save(save_path+experiment)
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    return model.evaluate(images_test, labels_test)
