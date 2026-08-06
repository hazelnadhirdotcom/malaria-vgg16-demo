import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import sys
import gdown

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import terapkan_clahe, terapkan_gamma_correction
import config
from tensorflow.keras.models import load_model

# ======================================================
# KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="Deteksi Malaria - VGG16",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# CSS CUSTOM - TEMA BIRU-HIJAU MEDIS
# ======================================================
st.markdown("""
<style>
    /* Warna dasar aplikasi */
    .stApp {
        background-color: #f4f9f9;
    }

    /* Header utama */
    .header-container {
        background: linear-gradient(90deg, #1e3a8a 0%, #2563eb 100%);
        padding: 28px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(30, 58, 138, 0.25);
    }
    .header-container h1 {
        color: white;
        margin: 0;
        font-size: 30px;
    }
    .header-container p {
        color: #dbeafe;
        margin: 6px 0 0 0;
        font-size: 15px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #eff6ff;
    }
    section[data-testid="stSidebar"] * {
        color: #1e293b !important;
    }
    section[data-testid="stSidebar"] h3 {
        color: #1e3a8a !important;
    }
    section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #1e3a8a !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #bfdbfe;
    }

    /* Kotak upload */
    .stFileUploader {
        border-radius: 12px;
    }

    /* Tombol utama */
    .stButton>button {
        background: linear-gradient(90deg, #1e3a8a 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        width: 100%;
        transition: 0.2s;
    }
    .stButton>button:hover {
        opacity: 0.9;
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.35);
    }

    /* Kartu hasil kustom */
    .hasil-card {
        padding: 22px;
        border-radius: 14px;
        margin-top: 16px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    }
    .hasil-sehat {
        background-color: #eff6ff;
        border-left: 6px solid #2563eb;
    }
    .hasil-infeksi {
        background-color: #fef2f2;
        border-left: 6px solid #dc2626;
    }
    .hasil-card h3 {
        margin: 0 0 6px 0;
    }
    .hasil-sehat h3 { color: #1e3a8a; }
    .hasil-infeksi h3 { color: #dc2626; }

    .footer-custom {
        text-align: center;
        color: #6b7280;
        font-size: 13px;
        margin-top: 40px;
        padding-top: 16px;
        border-top: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER
# ======================================================
st.markdown("""
<div class="header-container">
    <h1>🔬 Deteksi Malaria dari Citra Sel Darah</h1>
    <p>Klasifikasi otomatis sel darah menggunakan VGG-16 dengan preprocessing CLAHE & Gamma Correction</p>
</div>
""", unsafe_allow_html=True)

# ======================================================
# FILE ID GOOGLE DRIVE
# ======================================================
FILE_ID_MODEL_A = "1plXM5oMDKCiJ83R8YJDeQzt_T2zSs3AT"
FILE_ID_MODEL_B = "14KVXLMaHgeTNttWZKBIskLWyn1DYOzwR"
FOLDER_MODEL_LOKAL = "model_cache"
os.makedirs(FOLDER_MODEL_LOKAL, exist_ok=True)

@st.cache_resource
def download_dan_muat_model(file_id, nama_file):
    path_lokal = os.path.join(FOLDER_MODEL_LOKAL, nama_file)
    if not os.path.exists(path_lokal):
        url = f"https://drive.google.com/uc?id={file_id}"
        with st.spinner(f"Mengunduh model {nama_file} (sekali saja, mohon tunggu)..."):
            gdown.download(url, path_lokal, quiet=False)
    return load_model(path_lokal)

# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan Model")
    pilihan_model = st.radio(
        "Pilih model yang digunakan:",
        ["Model A (tanpa preprocessing)", "Model B (dengan CLAHE + Gamma Correction)"],
        index=1
    )

    st.markdown("---")
    st.markdown("### 📊 Info Model")
    if pilihan_model == "Model A (tanpa preprocessing)":
        st.metric("Akurasi Training", "93.03%")
        st.caption("Model dilatih langsung dari citra asli tanpa tahap preprocessing tambahan.")
    else:
        st.metric("Akurasi Training", "93.48%")
        st.caption("Model dilatih dari citra yang sudah melalui CLAHE (peningkatan kontras) dan Gamma Correction (koreksi pencahayaan).")

    st.markdown("---")
    st.caption("Dibangun dengan VGG-16 (Transfer Learning) · TensorFlow/Keras · Streamlit")

if pilihan_model == "Model A (tanpa preprocessing)":
    model = download_dan_muat_model(FILE_ID_MODEL_A, "model_a.keras")
else:
    model = download_dan_muat_model(FILE_ID_MODEL_B, "model_b.keras")

# ======================================================
# LAYOUT UTAMA - 2 KOLOM
# ======================================================
kol_kiri, kol_kanan = st.columns([1, 1], gap="large")

with kol_kiri:
    st.markdown("#### 📤 Upload Gambar")
    file_upload = st.file_uploader(
        "Upload gambar sel darah (jpg/png)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if file_upload is not None:
        gambar_pil = Image.open(file_upload).convert("RGB")
        st.image(gambar_pil, caption="Gambar yang diupload", use_container_width=True)

with kol_kanan:
    st.markdown("#### 🧪 Hasil Deteksi")

    if file_upload is None:
        st.info("Upload gambar sel darah di sebelah kiri untuk memulai deteksi.")
    else:
        gambar_np = np.array(gambar_pil)
        gambar_bgr = cv2.cvtColor(gambar_np, cv2.COLOR_RGB2BGR)

        if pilihan_model != "Model A (tanpa preprocessing)":
            gambar_bgr = terapkan_clahe(gambar_bgr)
            gambar_bgr = terapkan_gamma_correction(gambar_bgr, gamma=1.5)

        gambar_resize = cv2.resize(gambar_bgr, config.IMAGE_SIZE)
        gambar_final = gambar_resize.astype("float32") / 255.0
        gambar_final = np.expand_dims(gambar_final, axis=0)

        if st.button("🔍 Deteksi Sekarang"):
            with st.spinner("Menganalisis citra..."):
                prediksi = model.predict(gambar_final)[0][0]

            if prediksi > 0.5:
                st.markdown(f"""
                <div class="hasil-card hasil-sehat">
                    <h3>✅ UNINFECTED (Sehat)</h3>
                    <p>Tingkat keyakinan model: <b>{prediksi*100:.1f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="hasil-card hasil-infeksi">
                    <h3>⚠️ PARASITIZED (Terinfeksi Malaria)</h3>
                    <p>Tingkat keyakinan model: <b>{(1-prediksi)*100:.1f}%</b></p>
                </div>
                """, unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
<div class="footer-custom">
    Dikembangkan untuk keperluan penelitian & kompetisi · Model: VGG-16 Transfer Learning<br>
    ⚠️ Hasil deteksi ini bersifat prediktif dan tidak menggantikan diagnosis medis profesional.
</div>
""", unsafe_allow_html=True)