import time
import tkinter as tk

import cv2
import numpy as np

import my_face

# decrease it with good hardware
FRAME_DELAY_MS = 10


class Example(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)

        # create a prompt, an input box, an output label,
        # and a button to do the computation
        self.prompt = tk.Label(self, text="Threshold:", anchor="w")
        self.entry = tk.Entry(self, text="0.7")
        self.start = tk.Button(self, text="Start", command=self.start)
        self.capture_target = tk.Button(self, text="Capture Target", command=self.check_face)

        # lay the widgets out on the screen.
        self.prompt.pack(side="top", fill="x")
        self.entry.pack(side="top", fill="x", padx=20)
        self.start.pack(side="right")
        self.capture_target.pack(side="right")

        self.video_capture = cv2.VideoCapture(0)
        self.face_recognition = my_face.Recognition()
        self.target_face_embedding = None
        self.capture_target_face = False
        self.unlocked = False
        self.threshold = 0

    def check_face(self):
        try:
            self.threshold = float(self.entry.get())
        except ValueError:
            pass
        print("Set threshold: " + str(self.threshold))
        if self.threshold > 0:
            self.capture_target_face = True

    def start(self):
        self.video_loop()
        # self.video_capture.release()
        # cv2.destroyAllWindows()

    def video_loop(self):
        ok, frame = self.video_capture.read()
        if ok:
            key = cv2.waitKey(FRAME_DELAY_MS)
            # 从frame上识别人脸, 人名和bounding box, frame上可能包括多个人脸
            faces = self.face_recognition.identify(frame)
            self.add_overlays(frame, faces)
            cv2.imshow('Video', frame)

        self.after(FRAME_DELAY_MS, self.video_loop)

    def add_overlays(self, frame, faces):
        if faces is not None:
            color = (0, 0, 255)
            if self.unlocked:
                color = (0, 255, 0)

            for face in faces:
                face_bb = face.bounding_box.astype(int)
                cv2.rectangle(frame,
                              (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                              color, 2)

                if face.embedding is not None:
                    if self.capture_target_face:
                        self.target_face_embedding = face.embedding
                        print("target face: ", str(face.embedding))
                        self.capture_target_face = False

                    if self.target_face_embedding is not None:
                        dist = np.sqrt(np.sum(np.square(np.subtract(self.target_face_embedding, face.embedding))))
                        if dist < self.threshold:
                            self.unlocked = True
                        else:
                            self.unlocked = False

                        cv2.putText(frame, str(dist), (face_bb[0], face_bb[3]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                                    thickness=2, lineType=2)


# if this is run as a program (versus being imported),
# create a root window and an instance of our example,
# then start the event loop

if __name__ == "__main__":
    root = tk.Tk()
    Example(root).pack(fill="both", expand=True)
    root.mainloop()
