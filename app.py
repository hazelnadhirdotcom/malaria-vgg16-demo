import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import sys
import gdown  # BARU

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import terapkan_clahe, terapkan_gamma_correction
import config
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Deteksi Malaria", page_icon="🔬")

st.title("🔬 Deteksi Malaria dari Citra Sel Darah")
st.write("Upload gambar sel darah, sistem akan mendeteksi apakah terinfeksi parasit malaria atau tidak.")

# BARU: File ID Google Drive untuk tiap model
FILE_ID_MODEL_A = "1plXM5oMDKCiJ83R8YJDeQzt_T2zSs3AT"
FILE_ID_MODEL_B = "14KVXLMaHgeTNttWZKBIskLWyn1DYOzwR"

# BARU: folder sementara buat nyimpen model yang didownload
FOLDER_MODEL_LOKAL = "model_cache"
os.makedirs(FOLDER_MODEL_LOKAL, exist_ok=True)

# Pilihan model
pilihan_model = st.radio(
    "Pilih model yang digunakan:",
    ["Model A (tanpa preprocessing)", "Model B (dengan CLAHE + Gamma Correction)"]
)

# BARU: fungsi download model dari Google Drive kalau belum ada di lokal
@st.cache_resource
def download_dan_muat_model(file_id, nama_file):
    path_lokal = os.path.join(FOLDER_MODEL_LOKAL, nama_file)
    if not os.path.exists(path_lokal):
        url = f"https://drive.google.com/uc?id={file_id}"
        with st.spinner(f"Mengunduh model {nama_file} (sekali saja, mohon tunggu)..."):
            gdown.download(url, path_lokal, quiet=False)
    return load_model(path_lokal)

if pilihan_model == "Model A (tanpa preprocessing)":
    model = download_dan_muat_model(FILE_ID_MODEL_A, "model_a.keras")
else:
    model = download_dan_muat_model(FILE_ID_MODEL_B, "model_b.keras")

file_upload = st.file_uploader("Upload gambar sel darah (jpg/png)", type=["jpg", "jpeg", "png"])

if file_upload is not None:
    gambar_pil = Image.open(file_upload).convert("RGB")
    st.image(gambar_pil, caption="Gambar yang diupload", width=300)

    gambar_np = np.array(gambar_pil)
    gambar_bgr = cv2.cvtColor(gambar_np, cv2.COLOR_RGB2BGR)

    if pilihan_model != "Model A (tanpa preprocessing)":
        gambar_bgr = terapkan_clahe(gambar_bgr)
        gambar_bgr = terapkan_gamma_correction(gambar_bgr, gamma=1.5)

    gambar_resize = cv2.resize(gambar_bgr, config.IMAGE_SIZE)
    gambar_final = gambar_resize.astype("float32") / 255.0
    gambar_final = np.expand_dims(gambar_final, axis=0)

    if st.button("Deteksi Sekarang"):
        prediksi = model.predict(gambar_final)[0][0]

        if prediksi > 0.5:
            st.success(f"✅ Hasil: UNINFECTED (Sehat) — keyakinan {prediksi*100:.1f}%")
        else:
            st.error(f"⚠️ Hasil: PARASITIZED (Terinfeksi Malaria) — keyakinan {(1-prediksi)*100:.1f}%")