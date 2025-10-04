import cv2
import os

def collect_data():
    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    face_id = input("Nhập ID người dùng (số): ")
    face_name = input("Nhập Tên người dùng: ")

    save_path = "database"
    os.makedirs(save_path, exist_ok=True)

    with open("labels.txt", "a", encoding="utf-8") as f:
        f.write(f"{face_id},{face_name}\n")

    count = 0
    print("Bắt đầu thu thập dữ liệu, nhấn Q để thoát...")
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite(f"{save_path}/User.{face_id}.{count}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Đã lưu {count} ảnh cho {face_name}")
