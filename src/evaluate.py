import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import config
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def buat_validation_generator_evaluasi():
    # Sengaja dipisah dari dataset_loader.py, karena buat evaluasi
    # urutan data HARUS tetap (shuffle=False), biar label dan prediksi nyambung benar
    datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
    val_gen = datagen.flow_from_directory(
        config.DATASET_DIR,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False   # <- kunci perbaikannya di sini
    )
    return val_gen

def evaluasi_model(path_model=None):
    if path_model is None:
        path_model = os.path.join(config.MODEL_OUTPUT_DIR, "model_terbaik.keras")

    print("Memuat model dari:", path_model)
    model = load_model(path_model)

    print("Memuat data validasi (tanpa acak urutan)...")
    val_gen = buat_validation_generator_evaluasi()

    print("Melakukan prediksi terhadap data validasi...")
    prediksi_prob = model.predict(val_gen, verbose=1)
    prediksi_label = (prediksi_prob > 0.5).astype(int).flatten()

    label_asli = val_gen.classes
    nama_kelas = list(val_gen.class_indices.keys())

    print("\n=== Classification Report ===")
    print(classification_report(label_asli, prediksi_label, target_names=nama_kelas))

    cm = confusion_matrix(label_asli, prediksi_label)
    print("\n=== Confusion Matrix ===")
    print(cm)

    os.makedirs(config.PLOTS_OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=nama_kelas, yticklabels=nama_kelas)
    plt.xlabel('Prediksi Model')
    plt.ylabel('Label Sebenarnya')
    plt.title('Confusion Matrix - Deteksi Malaria')
    path_simpan = os.path.join(config.PLOTS_OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(path_simpan)
    print("\nConfusion matrix disimpan di:", path_simpan)

if __name__ == "__main__":
    evaluasi_model()