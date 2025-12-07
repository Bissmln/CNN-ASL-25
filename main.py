import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="ASL Detection",
    page_icon="🖐️",
    layout="centered"
)

# --- 1. LOAD MODEL ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model_asl_huruf.h5')
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error memuat model: {e}. Pastikan file .h5 sudah di-upload.")

# --- 2. DEFINISI KELAS ---
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
    'del', 'nothing', 'space'
]

# --- 3. FUNGSI PREDIKSI ---
def import_and_predict(image_data, model):
    size = (64, 64)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    
    if img.shape[-1] == 4:
        img = img[:,:,:3]
        
    img = img / 255.0
    img_reshape = np.expand_dims(img, axis=0)
    prediction = model.predict(img_reshape)
    return prediction

# --- 4. TAMPILAN UTAMA ---
st.title("🖐️ Deteksi Bahasa Isyarat (ASL)")
st.write("Aplikasi ini menggunakan CNN untuk mendeteksi huruf ASL dari kamera.")

# Sidebar (Tanpa Logo)
with st.sidebar:
    st.header("Panduan Penggunaan")
    st.write("1. Pastikan cahaya ruangan cukup terang.")
    st.write("2. Gunakan background polos (tembok).")
    st.write("3. Posisikan tangan agar terlihat jelas di kamera.")

# --- 5. INPUT KAMERA ---
cam_input = st.camera_input("Ambil Foto Tangan Anda")

if cam_input is not None:
    image = Image.open(cam_input)
    
    # Lakukan Prediksi
    with st.spinner('Sedang memproses...'):
        predictions = import_and_predict(image, model)
        result_index = np.argmax(predictions)
        result_label = class_names[result_index]
        probability = np.max(predictions)

    # --- 6. TAMPILKAN HASIL ---
    st.write("---")
    st.subheader("Hasil Prediksi:")

    THRESHOLD = 0.70
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(image, caption="Foto Input", width=150)
        
    with col2:
        if probability > THRESHOLD:
            # Menampilkan Huruf yang terdeteksi dengan font besar
            st.markdown(f"<h1 style='text-align: left; color: #4CAF50;'>{result_label}</h1>", unsafe_allow_html=True)
            st.success(f"Akurasi/Keyakinan: {probability*100:.2f}%")
            
            if result_label == 'space':
                st.info("Simbol: SPASI")
            elif result_label == 'del':
                st.info("Simbol: HAPUS")
        else:
            st.warning("⚠️ Model kurang yakin.")
            st.write(f"Prediksi terdekat: **{result_label}** ({probability*100:.2f}%)")
            st.write("Saran: Coba posisikan tangan lebih dekat atau cari cahaya lebih terang.")

else:
    st.info("Silakan izinkan akses kamera dan klik 'Take Photo' untuk memulai.")