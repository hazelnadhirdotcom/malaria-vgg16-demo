import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import config

def bangun_model():
    # Ambil VGG-16 yang udah dilatih di ImageNet, tanpa lapisan klasifikasi aslinya
    # (include_top=False artinya kita buang bagian "kepala" yang lama)
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3)
    )

    # Bekukan semua layer VGG-16 asli, biar "pengetahuan" yang udah dipelajari
    # nggak berubah/rusak pas training kita nanti
    for layer in base_model.layers:
        layer.trainable = False

    # Tambahkan "kepala" baru khusus buat klasifikasi malaria
    x = base_model.output
    x = Flatten()(x)                          # ubah dari bentuk gambar jadi 1 baris angka panjang
    x = Dense(256, activation='relu')(x)       # layer belajar pola khusus dataset kita
    x = Dropout(0.5)(x)                        # matikan 50% neuron secara acak tiap training, biar model gak "menghafal" (overfitting)
    output = Dense(1, activation='sigmoid')(x) # 1 angka output: 0 = Parasitized, 1 = Uninfected

    model = Model(inputs=base_model.input, outputs=output)

    # Compile = kasih tau model gimana cara "belajar"
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',   # cocok buat klasifikasi 2 kelas
        metrics=['accuracy']
    )

    return model

if __name__ == "__main__":
    model = bangun_model()
    model.summary()