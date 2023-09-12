from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton
from kivy.uix.image import Image
import cv2
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import mediapipe as mp
import time
import cvzone

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2, refine_landmarks=True)
face_drw = mp.solutions.drawing_utils
face_drw_styles = mp.solutions.drawing_styles
upper = 0
lower = 0
detected = 0
# Initialize a variable to track when the drowsy alert was last triggered
drowsy_alert_start_time = None
drowsy_alert_duration = 10  # 10 seconds
flag = True


def eye_aspect_ratio(eye):
    # Extract the coordinates of eye landmarks
    print(eye[1])
    (x1, y1, _) = eye[1]
    (x2, y2, _) = eye[2]
    (x0, y0, _) = eye[0]
    (x3, y3, _) = eye[3]

    # Calculate the Euclidean distances
    A = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    C = ((x0 - x3) ** 2 + (y0 - y3) ** 2) ** 0.5

    # The EAR Equation
    EAR = A / (2 * C)
    return EAR


class MainApp(MDApp):
    def build(self):
        layout = MDBoxLayout(orientation='vertical')
        self.image = Image()
        layout.add_widget(self.image)  # Add the Image widget to the layout
        layout.add_widget(MDRaisedButton(
            text="CLICK HERE",
            pos_hint={'center_x': .5, 'center_y': .5},
            size_hint=(None, None)))
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.load_video, 1.0 / 30.0)
        return layout

    def load_video(self, *args):
        global flag, start_time
        success, img = self.capture.read()
        if flag:
            start_time = time.time()
        self.image_frame = img
        imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        right_eye_list = []
        left_eye_list = []
        results = face_mesh.process(imgRgb)
        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
                left_eye = [33, 145, 159, 133]
                right_eye = [263, 374, 386, 362]

                for id_f, lm in enumerate(face.landmark):
                    h, w, c = img.shape
                    cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    for right_eye_lm in right_eye:
                        if id_f == right_eye_lm:
                            if id_f == 263:
                                right_eye_list.insert(0, [cx, cy, cz])
                            if id_f == 362:
                                right_eye_list.insert(-1, [cx, cy, cz])
                            if id_f == 374:
                                right_eye_list.insert(1, [cx, cy, cz])
                            if id_f == 386:
                                right_eye_list.insert(2, [cx, cy, cz])
                            cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
                    for left_eye_lm in left_eye:
                        if id_f == left_eye_lm:
                            if id_f == 33:
                                left_eye_list.insert(0, [cx, cy, cz])
                            if id_f == 133:
                                left_eye_list.insert(-1, [cx, cy, cz])
                            if id_f == 145:
                                left_eye_list.insert(1, [cx, cy, cz])
                            if id_f == 159:
                                left_eye_list.insert(2, [cx, cy, cz])
                            cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
                    if len(right_eye_list) == 4 and len(left_eye_list) == 4:
                        right_ear = eye_aspect_ratio(right_eye_list)
                        left_ear = eye_aspect_ratio(left_eye_list)
                        if right_ear > 0.12 and left_ear > 0.12:
                            flag = True
                        else:
                            flag = False
                            end_time = time.time()
                            if (end_time - start_time) >= 2.5:
                                cvzone.putTextRect(img, f"DROWSY ALERT", (325, 40), scale=1.5,
                                                   offset=10, thickness=2, colorT=(39, 40, 41), colorR=(248, 222, 34))
                                flag = True
        buffer = cv2.flip(img, 0).tostring()
        texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture


if __name__ == "__main__":
    MainApp().run()
