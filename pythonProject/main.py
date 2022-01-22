import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import actions
import train
import utils


def draw_landmarks(hand_landmarks, image):
    mp.solutions.drawing_utils.draw_landmarks(
        image,
        hand_landmarks,
        mp.solutions.hands.HAND_CONNECTIONS,
        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
        mp.solutions.drawing_styles.get_default_hand_connections_style())


def main():
    ca = actions.ChooseColorAction()
    da = actions.DrawAction(ca)
    ea = actions.EraseAction()
    sa = actions.SelectAction()
    na = actions.NoAction()
    ma = actions.MoveAction(sa)
    action_dict = {"draw": da, "erase": ea, "select": sa, "nothing": na, "move": ma, "choose": ca}
    labels = ["choose", "draw", "erase", "move", "nothing", "select"]

    model = load_model("models/model-3conv")

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

                    prediction = model(np.array([utils.extract_hand(hand_landmarks, image, False)]))
                    a = labels[np.argmax(prediction[0])]
                    print(a)

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
    main()
    # train.train("model-3conv", batch_size=64, epochs=5)
