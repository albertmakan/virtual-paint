import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import load_model

from model import create_model_small, create_model_large


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

    images_train, images_test, labels_train, labels_test \
        = train_test_split(images, labels, test_size=0.1)
    images_train, images_validation, labels_train, labels_validation \
        = train_test_split(images_train, labels_train, test_size=0.15)

    model = create_model_large(gray) if load_path is None else load_model(load_path)

    history = model.fit(images_train, labels_train, validation_data=(images_validation, labels_validation),
                        epochs=epochs, batch_size=batch_size, verbose=2)

    model.save(save_path+experiment)

    print(history.history.keys())
    fig, axs = plt.subplots(2)

    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="validation accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy plot")

    axs[1].plot(history.history["loss"], label="train loss")
    axs[1].plot(history.history["val_loss"], label="validation loss")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss plot")

    plt.show()

    test_loss, test_acc = model.evaluate(images_test, labels_test)
    print(test_loss, test_acc)


if __name__ == '__main__':
    train("model-large-gray.h5", batch_size=32, epochs=10, gray=True)
