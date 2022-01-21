import cv2
import mediapipe as mp
import numpy as np

import actions
import create_dataset


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
    create_dataset.main()
