import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config
from preprocessing import terapkan_clahe, terapkan_gamma_correction

def buat_data_generator():
    # ImageDataGenerator = "mesin" yang otomatis baca gambar dari folder,
    # ubah ukurannya, normalisasi nilai pixel, dan bagi jadi training/validation
    datagen = ImageDataGenerator(
        rescale=1.0/255,        # ubah nilai pixel dari 0-255 jadi 0-1 (biar model lebih gampang belajar)
        validation_split=0.2    # 20% data buat validasi, 80% buat training
    )

    train_generator = datagen.flow_from_directory(
        config.DATASET_DIR,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='binary',      # cuma 2 kelas: Parasitized vs Uninfected
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        config.DATASET_DIR,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    return train_generator, validation_generator

def fungsi_preprocessing_untuk_generator(gambar):
    # Fungsi ini dipanggil otomatis oleh Keras untuk SETIAP gambar
    # sebelum masuk ke model, saat mode "dengan CLAHE+Gamma" aktif
    gambar_uint8 = gambar.astype('uint8')
    gambar_clahe = terapkan_clahe(gambar_uint8)
    gambar_gamma = terapkan_gamma_correction(gambar_clahe, gamma=1.5)
    gambar_final = gambar_gamma.astype('float32') / 255.0
    return gambar_final

def buat_data_generator_dengan_preprocessing():
    # Sama seperti buat_data_generator(), tapi menerapkan CLAHE + Gamma Correction
    # ke setiap gambar sebelum training
    datagen = ImageDataGenerator(
        preprocessing_function=fungsi_preprocessing_untuk_generator,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        config.DATASET_DIR,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        config.DATASET_DIR,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    return train_generator, validation_generator
if __name__ == "__main__":
    train_gen, val_gen = buat_data_generator()
    print("Jumlah gambar training:", train_gen.samples)
    print("Jumlah gambar validation:", val_gen.samples)
    print("Nama kelas:", train_gen.class_indices)