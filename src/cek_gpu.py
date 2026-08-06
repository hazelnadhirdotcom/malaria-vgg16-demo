import tensorflow as tf

print("Versi TensorFlow:", tf.__version__)
print("Jumlah GPU terdeteksi:", len(tf.config.list_physical_devices('GPU')))
print("Semua device yang terdeteksi:")
for device in tf.config.list_physical_devices():
    print(" -", device)