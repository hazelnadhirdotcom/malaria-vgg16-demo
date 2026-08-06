import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import config
from preprocessing import terapkan_clahe, terapkan_gamma_correction
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def fungsi_preprocessing_untuk_generator(gambar):
    gambar_uint8 = gambar.astype('uint8')
    gambar_clahe = terapkan_clahe(gambar_uint8)
    gambar_gamma = terapkan_gamma_correction(gambar_clahe, gamma=1.5)
    gambar_final = gambar_gamma.astype('float32') / 255.0
    return gambar_final

def buat_validation_generator_evaluasi():
    datagen = ImageDataGenerator(
        preprocessing_function=fungsi_preprocessing_untuk_generator,
        validation_split=0.2
    )
    val_gen = datagen.flow_from_directory(
        config.DATASET_DIR,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )
    return val_gen

def evaluasi_model_b():
    folder_b = os.path.join(config.MODEL_OUTPUT_DIR, "model_b_preprocessing")
    path_model = os.path.join(folder_b, "model_terbaik.keras")

    print("Memuat Model B dari:", path_model)
    model = load_model(path_model)

    print("Memuat data validasi (dengan CLAHE+Gamma, tanpa acak urutan)...")
    val_gen = buat_validation_generator_evaluasi()

    print("Melakukan prediksi...")
    prediksi_prob = model.predict(val_gen, verbose=1)
    prediksi_label = (prediksi_prob > 0.5).astype(int).flatten()

    label_asli = val_gen.classes
    nama_kelas = list(val_gen.class_indices.keys())

    print("\n=== Classification Report - Model B (dengan CLAHE+Gamma) ===")
    print(classification_report(label_asli, prediksi_label, target_names=nama_kelas))

    cm = confusion_matrix(label_asli, prediksi_label)
    print("\n=== Confusion Matrix - Model B ===")
    print(cm)

    os.makedirs(config.PLOTS_OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=nama_kelas, yticklabels=nama_kelas)
    plt.xlabel('Prediksi Model')
    plt.ylabel('Label Sebenarnya')
    plt.title('Confusion Matrix - Model B (CLAHE + Gamma Correction)')
    path_simpan = os.path.join(config.PLOTS_OUTPUT_DIR, 'confusion_matrix_model_b.png')
    plt.savefig(path_simpan)
    print("\nConfusion matrix Model B disimpan di:", path_simpan)

if __name__ == "__main__":
    evaluasi_model_b()