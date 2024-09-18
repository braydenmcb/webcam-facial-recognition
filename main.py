import cv2
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace
# https://www.geeksforgeeks.org/facial-expression-detection-using-deepface-module-in-python/     for facial expression detection
# python3 -m pip install opencv-python-headless numpy deepface matplotlib                        for the venv
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        cv2.putText(img, "No Face Found", (50,120), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,255), 10)
        return img
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
        result = DeepFace.analyze(img, actions = ['emotion'], enforce_detection=False)
        result = result[0]

        emotion = result['dominant_emotion']
        confidence = result['emotion'][emotion]
        formatted_conf = "{:.{}f}".format(confidence, 4)
        expression = emotion + " | " + formatted_conf + "% " + "confidence"

        cv2.putText(img, expression, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,173), 2)
    return img

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('Quick Face Detector', detect_faces(frame))
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()

