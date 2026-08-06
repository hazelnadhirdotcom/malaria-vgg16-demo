import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import config

def buat_grafik_model_b():
    folder_b = os.path.join(config.MODEL_OUTPUT_DIR, "model_b_preprocessing")
    path_csv = os.path.join(folder_b, "riwayat_training.csv")
    data = pd.read_csv(path_csv)

    print("Data Model B:")
    print(data)

    os.makedirs(config.PLOTS_OUTPUT_DIR, exist_ok=True)

    # Grafik akurasi Model B
    plt.figure(figsize=(10, 5))
    plt.plot(data['epoch'] + 1, data['accuracy'], marker='o', label='Akurasi Training')
    plt.plot(data['epoch'] + 1, data['val_accuracy'], marker='o', label='Akurasi Validasi')
    plt.xlabel('Epoch')
    plt.ylabel('Akurasi')
    plt.title('Grafik Akurasi Model B (dengan CLAHE + Gamma Correction)')
    plt.legend()
    plt.grid(True)
    path_akurasi = os.path.join(config.PLOTS_OUTPUT_DIR, 'grafik_akurasi_model_b.png')
    plt.savefig(path_akurasi)
    print("Grafik akurasi Model B disimpan di:", path_akurasi)

    # Grafik loss Model B
    plt.figure(figsize=(10, 5))
    plt.plot(data['epoch'] + 1, data['loss'], marker='o', label='Loss Training')
    plt.plot(data['epoch'] + 1, data['val_loss'], marker='o', label='Loss Validasi')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Grafik Loss Model B (dengan CLAHE + Gamma Correction)')
    plt.legend()
    plt.grid(True)
    path_loss = os.path.join(config.PLOTS_OUTPUT_DIR, 'grafik_loss_model_b.png')
    plt.savefig(path_loss)
    print("Grafik loss Model B disimpan di:", path_loss)

    # BONUS: Grafik perbandingan langsung Model A vs Model B (akurasi validasi)
    path_csv_a = os.path.join(config.MODEL_OUTPUT_DIR, "riwayat_training.csv")
    if os.path.exists(path_csv_a):
        data_a = pd.read_csv(path_csv_a)
        plt.figure(figsize=(10, 5))
        plt.plot(data_a['epoch'] + 1, data_a['val_accuracy'], marker='o', label='Model A (tanpa preprocessing)')
        plt.plot(data['epoch'] + 1, data['val_accuracy'], marker='o', label='Model B (CLAHE + Gamma)')
        plt.xlabel('Epoch')
        plt.ylabel('Akurasi Validasi')
        plt.title('Perbandingan Akurasi Validasi: Model A vs Model B')
        plt.legend()
        plt.grid(True)
        path_perbandingan = os.path.join(config.PLOTS_OUTPUT_DIR, 'grafik_perbandingan_a_vs_b.png')
        plt.savefig(path_perbandingan)
        print("Grafik perbandingan A vs B disimpan di:", path_perbandingan)

if __name__ == "__main__":
    buat_grafik_model_b()