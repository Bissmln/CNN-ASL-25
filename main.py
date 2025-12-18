import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# --- 1. KONFIGURASI HALAMAN (Modern & Wide) ---
st.set_page_config(
    page_title="ASL Real-time Detector",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS (Untuk UI Cantik) ---
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
    }
    .big-letter {
        font-size: 80px;
        font-weight: 800;
        color: #2e7d32;
        margin: 0;
        line-height: 1;
    }
    .confidence-text {
        font-size: 18px;
        color: #666;
        margin-bottom: 10px;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOAD MODEL (Cached) ---
@st.cache_resource
def load_model():
    # Pastikan file model ada di folder yang sama
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
    
    # Handle jika gambar punya 4 channel (RGBA)
    if img.shape[-1] == 4:
        img = img[:,:,:3]
        
    img = img / 255.0
    img_reshape = np.expand_dims(img, axis=0)
    prediction = model.predict(img_reshape)
    return prediction

# --- 6. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2088/2088617.png", width=80)
    st.title("Panduan")
    st.info("Aplikasi ini mendeteksi Bahasa Isyarat Amerika (ASL) secara real-time dari foto.")
    st.markdown("""
    **Tips Akurasi Tinggi:**
    1. 💡 Pastikan **cahaya terang**.
    2. 🧱 Gunakan **background polos**.
    3. ✋ Tangan harus terlihat **jelas**.
    """)
    st.write("---")
    st.caption("Developed for Computer Vision Project")

# --- 7. TAMPILAN UTAMA ---
st.title("🤟 Deteksi Bahasa Isyarat (ASL)")
st.write("Ambil foto gerakan tangan Anda menggunakan kamera di bawah ini.")
st.write("---")

# Layout Utama
col_cam, col_res = st.columns([1.5, 1])

with col_cam:
    st.subheader("📷 Kamera")
    cam_input = st.camera_input("Klik tombol untuk mengambil foto")

with col_res:
    st.subheader("📊 Hasil Deteksi")
    
    if cam_input is None:
        # Tampilan placeholder jika belum ada foto
        st.markdown("""
        <div class="result-card">
            <h3 style="color: #ccc;">Menunggu Foto...</h3>
            <p style="color: #999;">Silakan ambil foto di sebelah kiri.</p>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        # Jika foto diambil, lakukan prediksi
        image = Image.open(cam_input)
        
        with st.spinner('Menganalisis gambar...'):
            predictions = import_and_predict(image, model)
            result_index = np.argmax(predictions)
            result_label = class_names[result_index]
            probability = np.max(predictions)
            
        # Treshold keyakinan
        THRESHOLD = 0.60 

        if probability > THRESHOLD:
            # Tampilan Hasil Sukses (Card UI)
            
            # Label khusus
            display_text = result_label
            if result_label == 'space':
                display_text = "SPASI"
            elif result_label == 'del':
                display_text = "HAPUS"
            elif result_label == 'nothing':
                display_text = "-"

            st.markdown(f"""
            <div class="result-card">
                <p class="confidence-text">Terdeteksi sebagai:</p>
                <p class="big-letter">{display_text}</p>
                <p style="margin-top: 10px; font-weight: bold;">Akurasi: {probability*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar visual
            st.progress(int(probability * 100))
            st.success("✅ Prediksi Valid")
            
        else:
            # Tampilan jika ragu
            st.markdown(f"""
            <div class="result-card" style="border: 2px solid #ff9800;">
                <h2 style="color: #ff9800;">⚠️ Tidak Yakin</h2>
                <p>Mirip: <b>{result_label}</b> ({probability*100:.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)
            st.warning("Posisi tangan kurang jelas atau pencahayaan kurang. Silakan coba lagi.")

# --- FOOTER ---
if cam_input is not None:
    with st.expander("Lihat Detail Gambar Input"):
        st.image(image, caption="Gambar yang diproses model", width=200)