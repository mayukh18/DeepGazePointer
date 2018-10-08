import numpy as np
import cv2
import win32gui
from keras.models import load_model
from keras import backend as K
import string
import random
import pygame
from collections import deque

def euc_dist_keras(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True))

def new_euc_dist(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True)) + K.mean(K.abs(y_true - y_pred), axis=-1, keepdims=True)


def uniform_weight_euc(y_true, y_pred):
    n = K.shape(y_true)[0]
    xmid = K.constant(value=np.array([767]))
    ymid = K.constant(value=np.array([431]))
    ones = K.constant(value=np.array([1.]))

    a = K.reshape(3 * K.abs(y_true[:, 0] - xmid) / 1535. + ones, (n, 1))
    b = K.reshape(3 * K.abs(y_true[:, 1] - ymid) / 863. + ones, (n, 1))
    weights = K.concatenate([a, b], axis=1)
    return K.sqrt(K.sum(K.square(y_true - y_pred) * weights, axis=-1, keepdims=True))

def id_generator(size=7, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def skew(x, y):
    if x <= 400:
        x = 400 - (400-x)*1.8
    if y > 500:
        y = 500 + (y - 500)*1.5

    x = max(0, min(x, 1535))
    y = max(0, min(y, 863))
    return int(x), int(y)

class EyeCapture:
    FRONT_FACE_PATH = "./weights/front_face_haarcascade.xml"
    PROFILE_FACE_PATH = "./weights/profile_face_haarcascade.xml"
    EYE_PATH = "./weights/eye_haarcascade.xml"
    EYE_GLASSES_PATH = "./weights/eye_glasses_haarcascade.xml"
    DATA_PATH = "./data/"

    def __init__(self):
        self.model = load_model("eye55.model", custom_objects={'new_euc_dist':new_euc_dist, 'euc_dist_keras':euc_dist_keras, 'uniform_weight_euc':uniform_weight_euc})
        #self.model = load_model("best_eye.model", custom_objects={'euc_dist_keras': euc_dist_keras})
        pygame.init()
        size = width, height = 1535, 863
        #self.screen = pygame.display.set_mode(size)
        self.screen = pygame.display.set_mode((1535, 863), pygame.FULLSCREEN)

    def loop(self):
        coords = [0,0]
        frontface_cascade = cv2.CascadeClassifier(self.FRONT_FACE_PATH)
        profileface_cascade = cv2.CascadeClassifier(self.PROFILE_FACE_PATH)
        eye_cascade = cv2.CascadeClassifier(self.EYE_PATH)
        eye_glasses_cascade = cv2.CascadeClassifier(self.EYE_GLASSES_PATH)

        history = [(0,0), (0,0), (0,0), (0,0), (0,0)]
        history = deque(history)

        cap = cv2.VideoCapture(0)
        for i in range(1000):
            flag = False
            ret, img = cap.read()
            #cur_x, cur_y = win32gui.GetCursorPos()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #print(cur_x, cur_y)
            faces = []
            # ------------------------- front --------------------------- #
            faces = frontface_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                flag = True
            # ------------------------- side 1 --------------------------- #
            """
            if flag == False:
                faces = profileface_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    print(2)
                    flag = True
            # ------------------------- side 2 --------------------------- #
            if flag == False:
                faces = profileface_cascade.detectMultiScale(np.fliplr(gray), 1.3, 5)
                if len(faces) > 0:
                    print(3)
                    flag = True
            """
            eye = []
            for (x, y, w, h) in faces[:1]:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]

                eyes = eye_cascade.detectMultiScale(roi_gray)
                j = 0
                # print("eyes",len(eyes))
                eye_x = []
                for (ex, ey, ew, eh) in eyes:
                    if j > 1:
                        break
                    equ = gray[y + ey:y + ey + eh, x + ex:x + ex + ew]
                    equ = cv2.equalizeHist(equ)
                    eye.append(cv2.resize(equ, (100, 100), interpolation=cv2.INTER_CUBIC))
                    eye_x.append(ex)
                    # cv2.rectangle(gray, (x+ex, y+ey), (x+ex + ew, y+ey + eh), (0, 255, 0), 2)
                    j += 1

            if len(eye) == 2:
                if eye_x[1] > eye_x[0]:
                    eyes = np.hstack((eye[0], eye[1]))
                else:
                    eyes = np.hstack((eye[1], eye[0]))

                # save_name = self.DATA_PATH + "IM_" + id_generator() + "_" + str(cur_x) + "_" + str(cur_y) + ".jpg"
                # cv2.imwrite(save_name, eyes)
                # cv2.imshow('eyes', eyes)
                eyes = np.reshape(eyes, (-1, 100, 200, 1))
                eye[0] = eyes[:, :, :100, :]
                eye[1] = eyes[:, :, 100:, :]
                out = self.model.predict([eye[0], eye[1]])
                #print(out)
                #out[0][0] = max(0, out[0][0])
                #out[0][1] = max(0, out[0][1])
                history.pop()
                history.appendleft(out[0])

            self.screen.fill([0, 0, 0])
            x = int((history[0][0]*5 + history[1][0]*4 + history[2][0]*3 + history[3][0]*2 + history[4][0])/15)
            y = int((history[0][1]*5 + history[1][1]*4 + history[2][1]*3 + history[3][1]*2 + history[4][1])/15)
            x, y = skew(x,y)
            print(x, y)
            pygame.draw.circle(self.screen, [255, 0, 0], [x, y], 10, 10)
            pygame.display.flip()

        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()


if __name__ == "__main__":
    e = EyeCapture()
    e.loop()


