import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model_klasifikasi_kurma.joblib")

st.set_page_config(
	page_title="Klasifikasi Kurma",
	page_icon=":palm_tree:"
)

st.title(":palm_tree: Klasifikasi Kurma")
st.markdown("Aplikasi machine learning untuk klasifikasi kurma bagus, sedang atau jelek")

ukuran_cm = st.slider("Ukuran (cm)", 0.1, 5.0, 3.5)
berat_g = st.slider("Berat (g)", 5.0, 25.0, 15.0)
kekerasan_N = st.slider("Kekerasan (N)", 5.0, 100.0, 47.0)
kadar_gula_brix = st.slider("Kadar Gula (Brix)", 15.0, 70.0, 35.0)
kadar_air_pct = st.slider("Kadar Air (%)", 5.0, 50.0, 20.0)
varietas = st.pills("Varietas", ["Ajwa", "Medjool", "Deglet Noor", "Zahidi", "Barhi"], default="Ajwa")
tingkat_warna = st.pills("Tingkat Warna", ["Kuning","Cokelat Muda","Cokelat Tua", "Hitam"], default="Kuning")
grade_cacat = st.pills("Grade Cacat", ["Tidak Ada","Ringan", "Sedang", "Berat"], default="Tidak Ada")

if st.button("Prediksi", type="primary"):
	data = pd.DataFrame(
		[[ukuran_cm, berat_g, kekerasan_N, kadar_gula_brix, kadar_air_pct, varietas, tingkat_warna, grade_cacat]], 
		columns=["ukuran_cm", "berat_g", "kekerasan_N", "kadar_gula_brix", "kadar_air_pct", "varietas", "tingkat_warna", "grade_cacat"]
	)
	
	prediksi = model.predict(data)[0]
	presentase = max(model.predict_proba(data)[0])
	st.success(f"Prediksi **{prediksi}** dengan tingkat keyakinan **{presentase*100:.2f}%**")
	st.balloons()

st.divider()
st.caption("Dibuat dengan :heart: oleh **Aldean Afgan**")