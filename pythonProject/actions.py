from abc import ABC, abstractmethod

import cv2
import numpy as np
from numpy import ndarray


class Action(ABC):
    @abstractmethod
    def execute(self, landmark_coordinates, canvases: list[ndarray], img):
        pass

    @abstractmethod
    def finish(self):
        pass


class NoAction(Action):
    def execute(self, landmark_coordinates, canvases: list[ndarray], img):
        pass

    def finish(self):
        pass


class ChooseColorAction(Action):
    def __init__(self):
        self.yp = 0
        self.palette = [(0, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (255, 0, 255)]
        self.color = self.palette[0]

    def execute(self, landmark_coordinates, canvases: list[ndarray], img):
        self.yp = landmark_coordinates[8][1]
        x1 = landmark_coordinates[20][0]
        y1 = landmark_coordinates[12][1]
        y2 = landmark_coordinates[2][1]
        h = (y2 - y1) // len(self.palette)
        for i, col in enumerate(self.palette):
            cv2.rectangle(img, (x1, y1 + i * h), (x1 + 100, y1 + (i + 1) * h), col, -1)
            if y1 + i * h <= self.yp < y1 + (i + 1) * h:
                self.color = col
                cv2.circle(img, landmark_coordinates[8], 3, self.color, 3)

    def finish(self):
        self.yp = 0


class DrawAction(Action):
    def __init__(self, color: ChooseColorAction):
        self.xp = 0
        self.yp = 0
        self.color_action = color
        self.canvases = []

    def execute(self, landmark_coordinates, canvases: list[ndarray], img):
        self.canvases = canvases
        xc, yc = landmark_coordinates[8]  # index finger tip
        cv2.circle(img, (xc, yc), 5, self.color_action.color, -1)
        if self.xp == 0 and self.yp == 0:
            self.xp = xc
            self.yp = yc
        cv2.line(canvases[1], (self.xp, self.yp), (xc, yc), self.color_action.color, 10)
        self.xp = xc
        self.yp = yc

    def finish(self):
        gray = cv2.cvtColor(self.canvases[1], cv2.COLOR_RGB2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        center, (h, w), angle = cv2.minAreaRect(np.squeeze(contours[0], axis=1))
        h = 0 if h < 20 else h - 20
        w = 0 if w < 20 else w - 20
        box = cv2.boxPoints((center, (h, w), angle))
        box = np.intp(box)

        gray1 = np.copy(gray)
        nz1 = np.count_nonzero(gray)
        cv2.drawContours(gray, [box], 0, 0, 10)
        nz2 = np.count_nonzero(gray)
        cv2.ellipse(gray1, (center, (h, w), angle), 0, 10)
        nz3 = np.count_nonzero(gray1)

        if nz3 < nz1 * 0.4:
            cv2.ellipse(self.canvases[0], (center, (h, w), angle), self.color_action.color, 10)
        elif nz2 < nz1 * 0.4:
            cv2.drawContours(self.canvases[0], [box], 0, self.color_action.color, 10)
        else:
            self.canvases[0] += self.canvases[1]
        self.canvases[1] -= self.canvases[1]
        self.xp = 0
        self.yp = 0


class EraseAction(Action):
    def __init__(self):
        self.xp = 0
        self.yp = 0

    def execute(self, landmark_coordinates, canvases: list[ndarray], img):
        xc, yc = landmark_coordinates[4]  # thumb tip
        cv2.circle(img, (xc, yc), 50, (0, 0, 0), -1)
        if self.xp == 0 and self.yp == 0:
            self.xp = xc
            self.yp = yc
        cv2.line(canvases[0], (self.xp, self.yp), (xc, yc), (0, 0, 0), 100)
        self.xp = xc
        self.yp = yc

    def finish(self):
        self.xp = 0
        self.yp = 0


class SelectAction(Action):
    def __init__(self):
        self.x1 = 0
        self.x2 = 0
        self.y1 = 0
        self.y2 = 0
        self.canvas = None

    def execute(self, landmark_coordinates, canvases: list[ndarray], img):
        self.canvas = canvases[2]
        self.x1 = landmark_coordinates[4][0]
        self.x2 = landmark_coordinates[20][0]
        if self.x1 > self.x2:
            self.x1, self.x2 = self.x2, self.x1
        self.y1 = landmark_coordinates[12][1]
        self.y2 = landmark_coordinates[0][1]
        cv2.rectangle(img, (self.x1, self.y1), (self.x2, self.y2), (255, 255, 255), 3)

    def finish(self):
        self.canvas[:] = 0
        cv2.rectangle(self.canvas, (self.x1, self.y1), (self.x2, self.y2), (255, 255, 255), 3)


class MoveAction(Action):
    def __init__(self, select: SelectAction):
        self.canvases = []
        self.xp = 0
        self.yp = 0
        self.area = None
        self.select_action = select

    def execute(self, landmark_coordinates, canvases: list[ndarray], img):
        self.canvases = canvases
        xc, yc = landmark_coordinates[20]
        print(self.select_action.x1, xc, self.select_action.x2, self.select_action.y1, yc, self.select_action.y2)
        if self.xp == 0 and self.yp == 0 and not (self.select_action.x1 < xc < self.select_action.x2
                                                  and self.select_action.y1 < yc < self.select_action.y2):
            print("AAAAAAAAAAAAA")
            return
        if self.area is None:
            self.area = canvases[0][self.select_action.y1:self.select_action.y2,
                                    self.select_action.x1:self.select_action.x2]
        h, w, _ = self.area.shape
        canvases[1][self.yp:self.yp + h, self.xp:self.xp + w] = 0
        ch, cw, _ = canvases[1].shape
        self.xp = xc if xc + w < cw else cw - w
        self.yp = yc if yc + h < ch else cw - h
        print(xc, yc)
        try:
            canvases[1][self.yp - h // 2:self.yp + h - h // 2, self.xp - w // 2:self.xp + w - w // 2] = self.area
        except ValueError as e:
            print(e)

    def finish(self):
        self.canvases[0] += self.canvases[1]
        self.canvases[1] -= self.canvases[1]
        self.xp = 0
        self.yp = 0
        self.area = None
