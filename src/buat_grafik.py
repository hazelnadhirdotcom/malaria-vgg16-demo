import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import config

def buat_grafik_training():
    path_csv = os.path.join(config.MODEL_OUTPUT_DIR, "riwayat_training.csv")
    data = pd.read_csv(path_csv)

    print("Data yang ditemukan:")
    print(data)

    os.makedirs(config.PLOTS_OUTPUT_DIR, exist_ok=True)

    # Grafik 1: Akurasi (training vs validation)
    plt.figure(figsize=(10, 5))
    plt.plot(data['epoch'] + 1, data['accuracy'], marker='o', label='Akurasi Training')
    plt.plot(data['epoch'] + 1, data['val_accuracy'], marker='o', label='Akurasi Validasi')
    plt.xlabel('Epoch')
    plt.ylabel('Akurasi')
    plt.title('Grafik Akurasi Model Selama Training')
    plt.legend()
    plt.grid(True)
    path_akurasi = os.path.join(config.PLOTS_OUTPUT_DIR, 'grafik_akurasi.png')
    plt.savefig(path_akurasi)
    print("Grafik akurasi disimpan di:", path_akurasi)

    # Grafik 2: Loss (training vs validation)
    plt.figure(figsize=(10, 5))
    plt.plot(data['epoch'] + 1, data['loss'], marker='o', label='Loss Training')
    plt.plot(data['epoch'] + 1, data['val_loss'], marker='o', label='Loss Validasi')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Grafik Loss Model Selama Training')
    plt.legend()
    plt.grid(True)
    path_loss = os.path.join(config.PLOTS_OUTPUT_DIR, 'grafik_loss.png')
    plt.savefig(path_loss)
    print("Grafik loss disimpan di:", path_loss)

if __name__ == "__main__":
    buat_grafik_training()