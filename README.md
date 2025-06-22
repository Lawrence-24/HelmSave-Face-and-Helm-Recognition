
# 🛡️ HelmSave - Face and Helmet Recognition

🚨 Sistem ini mendeteksi wajah pengguna dan mengecek apakah mereka memakai helm yang benar.

🔍 Teknologi yang digunakan:
- **YOLOv8**: Deteksi helm
- **MediaPipe**: Deteksi wajah
- **face_recognition**: Pengenalan wajah
- **CustomTkinter**: Tampilan antarmuka real-time

---

## 📦 Fitur Unggulan
- ✅ Deteksi helm & wajah real-time
- ✅ Cek apakah helm cocok dengan identitas
- ✅ GUI interaktif gelap
- ✅ Snapshot & log otomatis

---

## 🛠️ Instalasi Lengkap (Windows)

### 1. Clone Repositori
```bash
git clone https://github.com/username/Helmet-Face-Recognition.git
cd Helmet-Face-Recognition
```

### 2. Buat Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Install CMake dan Visual Studio Build Tools

❗ Jika mengalami error saat install `face_recognition`, biasanya karena `dlib`.  
Ikuti langkah ini:

✅ **CMake**  
Download dan install dari:  
👉 https://cmake.org/download/

✅ **Visual Studio Build Tools**  
Download dari:  
👉 https://visualstudio.microsoft.com/visual-cpp-build-tools/

📌 Saat instalasi:
- Pilih **Desktop Development with C++**
- Centang juga:
  - MSVC v143 atau terbaru
  - C++ CMake tools for Windows
  - Windows 11 SDK (atau sesuai versi Windows Anda)

---

### 4. Install Dlib Prebuilt (.whl)  
**(Wajib Python 3.10 karena proyek menggunakan versi ini)**

Download file:  
[dlib-19.22.99-cp310-cp310-win_amd64.whl](https://github.com/RPi-Distro/dlib-build/releases)

Install:
```bash
pip install dlib-19.22.99-cp310-cp310-win_amd64.whl
```

---

### 5. Install Dependensi Proyek
```bash
pip install face_recognition
pip install opencv-python mediapipe customtkinter Pillow ultralytics
```

---

## 🧠 Training YOLOv8 untuk Deteksi Helm

### Langkah 1: Buat Dataset di Roboflow
1. Kunjungi [Roboflow](https://roboflow.com/)
2. Buat **Project Baru** (Type: *Object Detection*)
3. Upload gambar helm dan labeli seperti:
   - `Helm "Namanya"`
   - `Helm "Namanya"`
   - dll.
4. Klik **"Generate"**
5. Pilih format export: **YOLOv8 PyTorch**
6. Klik tombol **Download**:
   - Pilih format: `YOLOv8 PyTorch`
   - Centang opsi ✅ **Show download code**
   - Salin kode Python yang muncul (akan digunakan di Colab)

---

### Langkah 2: Training di Google Colab
```python
from google.colab import drive
drive.mount('/content/drive')

!pip install roboflow ultralytics

from roboflow import Roboflow
rf = Roboflow(api_key="MASUKKAN_API_KEY_KAMU")
project = rf.workspace("nama-workspace").project("nama-project")
dataset = project.version(1).download("yolov8")
# Dataset akan berada di: /content/nama-folder-dataset/data.yaml

from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # Bisa ganti ke yolov8s.pt, yolov8m.pt, dll
model.train(data="/content/nama-folder-dataset/data.yaml", epochs=50, imgsz=640)

!cp /content/runs/detect/train/weights/best.pt /content/drive/MyDrive/HelmSave_Dataset/
# File best.pt akan tersimpan di Google Drive

---

### Langkah 3: Gunakan di Proyek
1. Simpan file `best.pt` di folder utama proyek
2. Edit baris berikut di kode:
```python
helmet_model = YOLO("best.pt")
```

---

## 🖼️ Mengganti Gambar Wajah

### Struktur Folder:
```
image/
├── "Nama Foto".jpg
├── "Nama Foto".jpg
├── "Nama Foto".jpg
└── "Nama Foto".jpg
```

### Langkah:
1. Masukkan foto wajah ke folder `image/`  
   - Format: `.jpg` atau `.png`
   - Syarat: Wajah depan & terang
2. Tambahkan ke kode:
```python
load_and_encode_image('image/Nama Foto.jpg', 'Label Namanya')
```

---

## 📸 Snapshot dan Log

- Klik tombol **Take Snapshot** di GUI
- Gambar akan tersimpan di folder `snapshots/`
- Semua aktivitas tercatat otomatis di panel log

---

## 🎥 Mode Kamera

### Default (webcam laptop):
```python
cap = cv2.VideoCapture(0)
```

### IP Camera:
```python
cap = cv2.VideoCapture('http://IP_ADDRESS:8080/?action=stream')
```

---

## ❓ FAQ

**Q: Wajah tidak dikenali?**  
➡️ Pastikan gambar wajah jelas & sudah ditambahkan ke folder + kode.

**Q: Helm tidak terdeteksi?**  
➡️ Cek kembali label class di Roboflow, contoh: `Helm Namanya`

---

## ✅ Selesai!

Selamat! Sistem deteksi wajah dan helm siap dijalankan 🎉

```bash
python main.py
```
