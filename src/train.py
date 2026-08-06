import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from dataset_loader import buat_data_generator
from model import bangun_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, CSVLogger
from tensorflow.keras.models import load_model

# File kecil buat nyimpen "sudah sampai epoch berapa"
FILE_PROGRESS = os.path.join(config.MODEL_OUTPUT_DIR, "progress_epoch.txt")

class SimpanProgressEpoch(Callback):
    # Callback custom: tiap 1 epoch selesai, catat nomor epoch-nya ke file kecil
    def on_epoch_end(self, epoch, logs=None):
        with open(FILE_PROGRESS, "w") as f:
            f.write(str(epoch + 1))

def main():
    print("=== Memuat dataset ===")
    train_gen, val_gen = buat_data_generator()
    print("Data training:", train_gen.samples, "| Data validasi:", val_gen.samples)

    os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)
    path_model_terbaik = os.path.join(config.MODEL_OUTPUT_DIR, "model_terbaik.keras")

    # Cek apakah ada training sebelumnya yang bisa dilanjutkan
    epoch_awal = 0
    if os.path.exists(path_model_terbaik) and os.path.exists(FILE_PROGRESS):
        print("=== Ditemukan progress sebelumnya, melanjutkan training ===")
        model = load_model(path_model_terbaik)
        with open(FILE_PROGRESS, "r") as f:
            epoch_awal = int(f.read().strip())
        print(f"Melanjutkan dari epoch {epoch_awal}")
    else:
        print("=== Membangun model VGG-16 baru ===")
        model = bangun_model()

    callbacks = [
        ModelCheckpoint(
            filepath=path_model_terbaik,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        SimpanProgressEpoch(),
        CSVLogger(os.path.join(config.MODEL_OUTPUT_DIR, "riwayat_training.csv"), append=True)
    ]

    print("=== Mulai/lanjut training ===")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS,
        initial_epoch=epoch_awal,
        callbacks=callbacks
    )

    print("=== Training selesai ===")
    model.save(os.path.join(config.MODEL_OUTPUT_DIR, "model_terakhir.keras"))
    print("Model tersimpan di folder:", config.MODEL_OUTPUT_DIR)

    return history

if __name__ == "__main__":
    main()