import cv2
import numpy as np

def terapkan_clahe(gambar):
    # CLAHE bekerja di ruang warna LAB, bukan RGB, makanya kita convert dulu
    lab = cv2.cvtColor(gambar, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Bikin objek CLAHE: clipLimit = batas penajaman, tileGridSize = ukuran area lokal
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    # Gabung lagi channel-nya, convert balik ke RGB
    lab_clahe = cv2.merge((l_clahe, a, b))
    hasil = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return hasil

def terapkan_gamma_correction(gambar, gamma=1.5):
    # Bikin "tabel konversi" nilai pixel 0-255 sesuai rumus gamma correction
    invGamma = 1.0 / gamma
    tabel = np.array([
        ((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)
    ]).astype("uint8")

    # Terapkan tabel itu ke setiap pixel gambar
    return cv2.LUT(gambar, tabel)

def preprocess_gambar(path_gambar):
    gambar = cv2.imread(path_gambar)
    gambar_clahe = terapkan_clahe(gambar)
    gambar_final = terapkan_gamma_correction(gambar_clahe, gamma=1.5)
    return gambar_final

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config

    # Ambil 1 contoh gambar dari folder Parasitized buat dites
    contoh_file = os.listdir(config.PARASITIZED_DIR)[0]
    path_contoh = os.path.join(config.PARASITIZED_DIR, contoh_file)

    print("Memproses gambar:", path_contoh)
    hasil = preprocess_gambar(path_contoh)
    print("Berhasil! Ukuran gambar hasil:", hasil.shape)

    # Simpan hasil buat dilihat visualnya
    cv2.imwrite("outputs/plots/contoh_hasil_preprocessing.jpg", hasil)
    print("Gambar hasil disimpan di outputs/plots/contoh_hasil_preprocessing.jpg")