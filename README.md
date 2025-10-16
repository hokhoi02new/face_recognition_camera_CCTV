# Face Recognition using OpenCV (LBPH + Haar Cascade)

## 📖 Introduction
This project build a **real-time face recognition system** using:
- **Haar Cascade** for face detection  
- **LBPH (Local Binary Pattern Histogram)** for training and recognition  

Workflow:
1. **Collect face data** from webcam (save images into `database/`)  
2. **Train LBPH model** from collected data (generate `trainer.yml`)  
3. **Real-time face recognition** from camera, displaying the name/id people and confidence

---

## ⚙️ Project Structure
```
Face recognition/
│── database/        # Collected face images
│── cascades/        # Detecter
│── trainer/         # Trained model file  
│   │── trainer.yml          
│── main.py          # Main menu
│── collect.py       # Gathering face data of person and save to database
│── train.py         # Train LBPH model
│── recognize.py     # Real-time face recognition
│── labels.txt       # Mapping ID → Person name
│── requirements.txt 
```

---

## ⚙️ Usage
### Clone the repository

```bash
git clone https://github.com/hokhoi02new/face_recognition_camera_CCTV.git
cd face_recognition_camera_CCTV
```
### Install dependencies
```bash
pip install -r requirements.txt
```

### Collect Data
Run:
```bash
python main.py
```
Choose `1` from the menu to collect face data of person 
- Enter **User ID (number)** and **Name**  
- System will capture 50 images and save into `database`  
- User info is stored in `labels.txt`  


### Train Model
Choose `2` in the menu.  
- Images from `database/` will be used to train LBPH  
- Model is saved as `trainer/trainer.yml`  


### Face Recognition
Choose `3` in the menu.  
- System opens webcam, detects faces, and shows **Name** + **Confidence (%)**  
- Press `Q` to exit  

---

## 📝 Notes
- Each person should have at least **50 images** for better accuracy.  
- You can add multiple users by collecting data with different IDs.  
- For managing users (view/edit/delete in `labels.txt`), you may add an extra script.  

---


