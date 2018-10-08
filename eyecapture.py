import numpy as np
import cv2
import win32gui


import string
import random
def id_generator(size=7, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

class EyeCapture:
    FRONT_FACE_PATH = "./weights/front_face_haarcascade.xml"
    PROFILE_FACE_PATH = "./weights/profile_face_haarcascade.xml"
    EYE_PATH = "./weights/eye_haarcascade.xml"
    EYE_GLASSES_PATH = "./weights/eye_glasses_haarcascade.xml"
    DATA_PATH = "./data/"
    def __init__(self):
        return

    def loop(self):
        frontface_cascade = cv2.CascadeClassifier(self.FRONT_FACE_PATH)
        profileface_cascade = cv2.CascadeClassifier(self.PROFILE_FACE_PATH)
        eye_cascade = cv2.CascadeClassifier(self.EYE_PATH)
        eye_glasses_cascade = cv2.CascadeClassifier(self.EYE_GLASSES_PATH)

        cap = cv2.VideoCapture(0)
        for i in range(1000):
            flag = False
            ret, img = cap.read()
            cur_x, cur_y = win32gui.GetCursorPos()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print(cur_x, cur_y)
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
                #print("eyes",len(eyes))
                eye_x = []
                for (ex, ey, ew, eh) in eyes:
                    if j > 1:
                        break
                    equ = gray[y+ey:y+ey+eh, x+ex:x+ex+ew]
                    equ = cv2.equalizeHist(equ)
                    eye.append(cv2.resize(equ, (100, 100), interpolation = cv2.INTER_CUBIC))
                    eye_x.append(ex)
                    #cv2.rectangle(gray, (x+ex, y+ey), (x+ex + ew, y+ey + eh), (0, 255, 0), 2)
                    j += 1

            if len(eye)== 2:
                if eye_x[1] > eye_x[0]:
                    eyes = np.hstack((eye[0], eye[1]))
                else:
                    eyes = np.hstack((eye[1], eye[0]))

                save_name = self.DATA_PATH + "IM_" + id_generator() + "_" + str(cur_x) + "_" + str(cur_y) + ".jpg"
                cv2.imwrite(save_name, eyes)
                cv2.imshow('eyes', eyes)


                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break
            else:
                dummy = np.zeros((100, 200))
                cv2.imshow('eyes', dummy)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break
            cv2.imshow('img', gray)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        cap.release()
        cv2.destroyAllWindows()




if __name__ == "__main__":
    e = EyeCapture()
    e.loop()


