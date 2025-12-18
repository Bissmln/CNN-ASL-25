import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# --- 1. KONFIGURASI HALAMAN (Hapus konfigurasi sidebar) ---
st.set_page_config(
    page_title="ASL Real-time Detector",
    page_icon="🤟",
    layout="wide"
)

# --- 2. CUSTOM CSS (Perbaikan Tampilan Teks) ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stCameraInput {
        border-radius: 15px;
        overflow: hidden;
        border: 2px solid #e0e0e0;
    }
    .result-card {
        padding: 25px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
        height: 100%; /* Agar tinggi kartu konsisten */
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    /* --- Perbaikan CSS Teks di Sini --- */
    .big-letter {
        font-size: 80px;
        font-weight: 800;
        color: #2e7d32;
        margin-top: 15px;    /* Memberi jarak atas */
        margin-bottom: 15px; /* Memberi jarak bawah */
        line-height: 1;
    }
    .confidence-text {
        font-size: 18px;
        color: #666;
        margin: 0; /* Reset margin */
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOAD MODEL (Cached) ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model_asl.h5')
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Error memuat model: {e}. Pastikan file 'model_asl_huruf.h5' sudah di-upload.")

# --- 4. DEFINISI KELAS ---
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
    'del', 'nothing', 'space'
]

# --- 5. FUNGSI PREDIKSI ---
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

# --- 6. TAMPILAN UTAMA (Tanpa Sidebar) ---
st.title("🤟 Deteksi Bahasa Isyarat (ASL)")
st.write("Aplikasi ini mendeteksi Bahasa Isyarat Amerika (ASL) secara real-time dari foto kamera.")

# --- Panduan Dipindah ke Atas (Menggunakan Expander agar rapi) ---
with st.expander("ℹ️ Panduan & Tips Akurasi Tinggi"):
    col_guide1, col_guide2 = st.columns([1, 3])
    with col_guide1:
        st.image("https://cdn-icons-png.flaticon.com/512/2088/2088617.png", width=100)
    with col_guide2:
        st.markdown("""
        **Tips agar prediksi akurat:**
        1.  💡 Pastikan **cahaya ruangan terang**.
        2.  🧱 Gunakan **background polos** (misal: tembok).
        3.  ✋ Posisikan tangan agar terlihat **jelas dan fokus** di kamera.
        """)

st.write("---")

# --- Layout Dua Kolom (Kamera & Hasil) ---
col_cam, col_res = st.columns([1.5, 1])

with col_cam:
    st.subheader("📷 Kamera")
    cam_input = st.camera_input("Klik tombol untuk mengambil foto")

with col_res:
    st.subheader("📊 Hasil Deteksi")
    
    if cam_input is None:
        # Placeholder sebelum foto diambil
        st.markdown("""
        <div class="result-card" style="background-color: #f0f2f6; border: 2px dashed #ccc;">
            <h3 style="color: #999;">Menunggu Foto...</h3>
            <p style="color: #aaa;">Silakan ambil foto di sebelah kiri.</p>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        # Proses Prediksi
        image = Image.open(cam_input)
        with st.spinner('Menganalisis gambar...'):
            predictions = import_and_predict(image, model)
            result_index = np.argmax(predictions)
            result_label = class_names[result_index]
            probability = np.max(predictions)
            
        THRESHOLD = 0.60 

        if probability > THRESHOLD:
            # Label khusus
            display_text = result_label
            if result_label == 'space': display_text = "SPASI"
            elif result_label == 'del': display_text = "HAPUS"
            elif result_label == 'nothing': display_text = "-"

            # Tampilan Hasil Sukses (CSS Sudah Diperbaiki)
            st.markdown(f"""
            <div class="result-card">
                <p class="confidence-text">Terdeteksi sebagai:</p>
                <p class="big-letter">{display_text}</p>
                <div>
                    <p style="margin-bottom: 5px; font-weight: bold;">Akurasi: {probability*100:.1f}%</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(int(probability * 100))
            st.success("✅ Prediksi Valid")
            
        else:
            # Tampilan jika ragu
            st.markdown(f"""
            <div class="result-card" style="border: 2px solid #ff9800;">
                <h2 style="color: #ff9800; margin-top: 0;">⚠️ Tidak Yakin</h2>
                <p style="font-size: 24px; font-weight: bold;">{result_label}?</p>
                <p>Keyakinan hanya {probability*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            st.warning("Posisi tangan atau pencahayaan kurang jelas. Coba lagi.")

# --- FOOTER ---
if cam_input is not None:
    with st.expander("Lihat Detail Gambar Input"):
        st.image(image, caption="Gambar yang diproses model", width=200)