import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import load_model

from model import create_model_3conv, create_model_5conv


def read_data(gray=False, max_n=1000):
    labels = []
    images = []
    for i, c in enumerate(["choose", "draw", "erase", "move", "nothing", "select"]):
        folder_path = "../dataset/" + c
        n = 0
        for file in os.listdir(folder_path):
            if n > max_n:
                break
            image = cv2.imread(f"{folder_path}/{file}")
            if gray and image.shape[2] != 1:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            images.append(image)
            labels.append(i)
            images.append(cv2.flip(image, 1))
            labels.append(i)
            n += 1
    images = np.array(images, dtype="uint8")
    labels = np.array(labels)
    return images, labels


def train(experiment: str, save_path: str = "models/", load_path: str = None, batch_size=32, epochs=10, gray=False):
    images, labels = read_data(gray)
    images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.15)
    model = create_model_5conv(gray) if load_path is None else load_model(load_path)
    history = model.fit(images_train, labels_train, epochs=epochs, batch_size=batch_size, verbose=2)
    model.save(save_path+experiment)

    print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

    return model.evaluate(images_test, labels_test)


if __name__ == '__main__':
    train("model-5conv-gray1", batch_size=32, epochs=5, gray=True, load_path="models/model-5conv-gray")
