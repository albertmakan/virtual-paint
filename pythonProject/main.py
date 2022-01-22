import cv2
import mediapipe as mp
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
import actions
import create_dataset

path = "../dataset/"
types = ["choose", "draw", "erase", "move", "nothing", "select"]
testSize = 0.15


def draw_landmarks(hand_landmarks, image):
    mp.solutions.drawing_utils.draw_landmarks(
        image,
        hand_landmarks,
        mp.solutions.hands.HAND_CONNECTIONS,
        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
        mp.solutions.drawing_styles.get_default_hand_connections_style())


def dummy_network(landmark_list):
    # to be replaced with CNN
    fingers = []
    tips = [4, 8, 12, 16, 20]
    if landmark_list[tips[0]][0] >= landmark_list[tips[0] - 1][0]:
        fingers.append(0)
    else:
        fingers.append(1)
    for i in range(1, 5):
        if landmark_list[tips[i]][1] >= landmark_list[tips[i] - 2][1]:
            fingers.append(0)
        else:
            fingers.append(1)
    if fingers[1] and not fingers[0] and not fingers[2] and not fingers[3] and not fingers[4]:
        return "draw"
    if fingers[0] and not fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
        return "erase"
    if fingers[0] and fingers[1] and fingers[2] and fingers[3] and fingers[4]:
        return "select"
    if fingers[4] and not fingers[1] and not fingers[2] and not fingers[3] and not fingers[0]:
        return "move"
    if fingers[4] and fingers[2] and fingers[3] and not fingers[0]:
        return "choose"
    return "nothing"


def readData():
    labels = []
    images = []
    for i, type in enumerate(types):
        folderPath = path + type
        files = os.listdir(folderPath)
        for file in files:
            labels.append(i)
            image = cv2.imread(folderPath + '/' + file)
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    imageNum = len(images)
    images = np.array(images, dtype="uint8")
    images = images.reshape(imageNum, 120, 120, 1)
    labels = np.array(labels)
    # print(images)
    # print()
    # print(labels)
    return images, labels


def modelCreation():
    cnnModel = Sequential()
    cnnModel.add(Conv2D(32, (5, 5), activation='relu', input_shape=(120, 120, 1)))
    cnnModel.add(MaxPooling2D((2, 2)))

    cnnModel.add(Conv2D(64, (3, 3), activation='relu'))
    cnnModel.add(MaxPooling2D((2, 2)))

    cnnModel.add(Conv2D(64, (3, 3), activation='relu'))
    cnnModel.add(MaxPooling2D((2, 2)))
    cnnModel.add(Flatten())

    cnnModel.add(Dense(128, activation='relu'))
    cnnModel.add(Dense(10, activation='softmax'))
    cnnModel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return cnnModel


def cnn():
    images, labels = readData()
    imagesTrain, imagesTest, labelsTrain, labelsTest = train_test_split(images, labels, test_size=testSize)
    model = modelCreation()
    model.fit(imagesTrain, labelsTrain, epochs=5, batch_size=64,
              verbose=2)
    return model.evaluate(imagesTest, labelsTest)

def main():
    ca = actions.ChooseColorAction()
    da = actions.DrawAction(ca)
    ea = actions.EraseAction()
    sa = actions.SelectAction()
    na = actions.NoAction()
    ma = actions.MoveAction(sa)
    action_dict = {"draw": da, "erase": ea, "select": sa, "nothing": na, "move": ma, "choose": ca}

    capture = cv2.VideoCapture(0)
    capture.set(3, 1280)
    capture.set(4, 720)

    with mp.solutions.hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        current_a = "nothing"
        canvases = [
            np.zeros((720, 1280, 3), np.uint8),  # main canvas
            np.zeros((720, 1280, 3), np.uint8),  # temporary canvas
            np.zeros((720, 1280, 3), np.uint8)   # select box canvas
        ]

        while capture.isOpened():
            success, image = capture.read()
            if not success:
                continue
            h, w, c = image.shape
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # draw_landmarks(hand_landmarks, image)

                    lm_coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

                    a = dummy_network(lm_coords)

                    if a != current_a:
                        action_dict[current_a].finish()
                        current_a = a
                    action_dict[current_a].execute(lm_coords, canvases, image)

            for canvas in canvases:
                image = cv2.bitwise_or(image, canvas)
            cv2.imshow('Virtual Paint', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    capture.release()


if __name__ == '__main__':
    #main()
    cnn()
