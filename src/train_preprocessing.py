import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from dataset_loader import buat_data_generator_dengan_preprocessing
from model import bangun_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

def main():
    print("=== Memuat dataset (DENGAN CLAHE + Gamma Correction) ===")
    train_gen, val_gen = buat_data_generator_dengan_preprocessing()
    print("Data training:", train_gen.samples, "| Data validasi:", val_gen.samples)

    print("=== Membangun model VGG-16 (Model B) ===")
    model = bangun_model()

    # Folder terpisah khusus buat Model B, biar gak menimpa Model A
    folder_output_b = os.path.join(config.MODEL_OUTPUT_DIR, "model_b_preprocessing")
    os.makedirs(folder_output_b, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(folder_output_b, "model_terbaik.keras"),
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
        CSVLogger(os.path.join(folder_output_b, "riwayat_training.csv"), append=True)
    ]

    print("=== Mulai training Model B ===")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS,
        callbacks=callbacks
    )

    print("=== Training Model B selesai ===")
    model.save(os.path.join(folder_output_b, "model_terakhir.keras"))
    print("Model B tersimpan di folder:", folder_output_b)

if __name__ == "__main__":
    main()