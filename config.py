import os

# Path dasar folder project (otomatis terdeteksi, gak perlu diubah manual)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Path ke folder dataset
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "cell_images", "cell_images")
PARASITIZED_DIR = os.path.join(DATASET_DIR, "Parasitized")
UNINFECTED_DIR = os.path.join(DATASET_DIR, "Uninfected")

# Path buat nyimpen hasil
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "models")
PLOTS_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "plots")

# Parameter gambar & training (sesuai VGG-16)
IMAGE_SIZE = (224, 224)   # VGG-16 butuh input 224x224 pixel
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001

# Print buat ngecek (nanti bisa dihapus)
if __name__ == "__main__":
    print("Folder Parasitized:", PARASITIZED_DIR)
    print("Folder Uninfected:", UNINFECTED_DIR)
    print("Ada berapa gambar Parasitized:", len(os.listdir(PARASITIZED_DIR)))
    print("Ada berapa gambar Uninfected:", len(os.listdir(UNINFECTED_DIR)))