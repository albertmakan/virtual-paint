import cv2


def extract_hand(hand_landmarks, image, show=True):
    h, w, c = image.shape
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    x1, y1 = int(min(xs) * w), int(min(ys) * h)
    x2, y2 = int(max(xs) * w), int(max(ys) * h)

    dx, dy = x2 - x1, y2 - y1
    if dy < dx:
        m = dx // 10
        x1, x2 = x1 - m, x2 + m
        d = dx + m + m - dy
        y1, y2 = y1 - d // 2, y2 + d // 2
    else:
        m = dy // 10
        y1, y2 = y1 - m, y2 + m
        d = dy + m + m - dx
        x1, x2 = x1 - d // 2, x2 + d // 2

    x1 = max(0, x1)
    y1 = max(0, y1)
    hand_img = cv2.resize(image[y1:y2, x1:x2], (120, 120))
    if show:
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.imshow('cropped', hand_img)

    return hand_img
