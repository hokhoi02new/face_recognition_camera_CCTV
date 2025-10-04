import cv2
import os

def recognize_face():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer/trainer.yml")
    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

    names = {}
    if os.path.exists("labels.txt"):
        with open("labels.txt", "r", encoding="utf-8") as f:
            for line in f.readlines():
                id, name = line.strip().split(",")
                names[int(id)] = name

    cap = cv2.VideoCapture(0)
    print("Bắt đầu nhận diện, nhấn Q để thoát...")

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            if confidence < 70:
                text = f"{names.get(id, 'Unknown')} ({round(100 - confidence)}%)"
            else:
                text = "Unknown"

            cv2.putText(frame, text, (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
