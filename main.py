import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import cv2

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="ASL Detection",
    page_icon="🖐️",
    layout="centered"
)

# --- 1. LOAD MODEL (DI-CACHE AGAR CEPAT) ---
@st.cache_resource
def load_model():
    # Pastikan file model 'model_asl_huruf.h5' ada di satu folder dengan main.py
    model = tf.keras.models.load_model('model_asl_huruf.h5')
    return model

try:
    model = load_model()
    st.success("Model berhasil dimuat!")
except Exception as e:
    st.error(f"Error memuat model: {e}. Pastikan file .h5 sudah di-upload.")

# --- 2. DEFINISI KELAS (URUTAN HARUS SAMA DENGAN TRAINING) ---
# Urutan ini berdasarkan alfabetis folder di dataset Kaggle ASL
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
    'del', 'nothing', 'space'
]

# --- 3. FUNGSI PREPROCESSING & PREDIKSI ---
def import_and_predict(image_data, model):
    # a. Resize gambar ke 64x64 (sesuai input model kita)
    size = (64, 64)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    
    # b. Ubah ke array numpy
    img = np.asarray(image)
    
    # c. Pastikan format RGB
    if img.shape[-1] == 4: # Jika ada alpha channel (PNG)
        img = img[:,:,:3]
        
    # d. Normalisasi (0-255 jadi 0-1)
    img = img / 255.0
    
    # e. Tambah dimensi batch (1, 64, 64, 3)
    img_reshape = np.expand_dims(img, axis=0)
    
    # f. Prediksi
    prediction = model.predict(img_reshape)
    return prediction

# --- 4. TAMPILAN UTAMA (UI) ---
st.title("🖐️ Deteksi Bahasa Isyarat (ASL)")
st.write("Aplikasi ini menggunakan CNN untuk mendeteksi huruf ASL dari kamera.")

# Sidebar untuk Info
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/d/d4/American_Sign_Language_ASL.svg", width=100)
    st.write("**Panduan:**")
    st.write("1. Pastikan cahaya cukup.")
    st.write("2. Gunakan background polos.")
    st.write("3. Tangan harus terlihat jelas.")

# --- 5. INPUT KAMERA ---
cam_input = st.camera_input("Ambil Foto Tangan Anda")

if cam_input is not None:
    # Tampilkan gambar yang diambil
    image = Image.open(cam_input)
    
    # Lakukan Prediksi
    with st.spinner('Sedang memproses...'):
        predictions = import_and_predict(image, model)
        
        # Ambil hasil probabilitas tertinggi
        score = tf.nn.softmax(predictions[0])
        result_index = np.argmax(predictions)
        result_label = class_names[result_index]
        probability = np.max(predictions)

    # --- 6. TAMPILKAN HASIL DENGAN THRESHOLD ---
    st.write("---")
    st.subheader("Hasil Prediksi:")

    # Logika Threshold (Misal 70%)
    THRESHOLD = 0.70
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(image, caption="Foto Input", width=150)
        
    with col2:
        if probability > THRESHOLD:
            st.metric(label="Huruf Terdeteksi", value=result_label)
            st.success(f"Akurasi/Keyakinan: {probability*100:.2f}%")
            
            # Info tambahan jika spasi/del
            if result_label == 'space':
                st.info("Simbol: SPASI")
            elif result_label == 'del':
                st.info("Simbol: HAPUS")
            elif result_label == 'nothing':
                st.info("Tidak ada tangan terdeteksi.")
        else:
            st.warning("⚠️ Model kurang yakin.")
            st.write(f"Prediksi terdekat: **{result_label}** ({probability*100:.2f}%)")
            st.write("Saran: Coba posisikan tangan lebih dekat atau cari cahaya lebih terang.")

    # Tampilkan grafik probabilitas (Opsional, biar keren buat laporan)
    st.write("---")
    if st.checkbox("Tampilkan Detail Grafik Probabilitas"):
        st.bar_chart(predictions[0])

else:
    st.info("Silakan izinkan akses kamera dan klik 'Take Photo' untuk memulai.")