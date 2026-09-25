from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Cluster Listing Airbnb NYC")

folder = Path(__file__).parent


@st.cache_resource
def load_model():
    scaler = joblib.load(folder / "scaler.joblib")
    kmeans = joblib.load(folder / "kmeans.joblib")
    profil = pd.read_csv(folder / "profil_cluster.csv", index_col="cluster")
    return scaler, kmeans, profil


scaler, kmeans, profil = load_model()

st.title("Cluster Listing Airbnb New York")
st.write("Masukkan data listing Airbnb untuk mengetahui listing tersebut termasuk ke cluster mana.")

col1, col2 = st.columns(2)
with col1:
    price = st.number_input("Harga per malam (USD)", min_value=10.0, value=125.0, step=5.0)
    bedrooms = st.number_input("Jumlah kamar tidur (0 = studio)", min_value=0, max_value=15, value=1)
with col2:
    minimum_nights = st.number_input("Minimum menginap (malam)", min_value=1, max_value=1250, value=30)
    availability = st.number_input("Hari tersedia dalam setahun", min_value=0, max_value=365, value=180)

if st.button("Prediksi Cluster"):
    fitur = ["price", "bedrooms", "minimum_nights", "availability_365"]
    kolom_log = ["price", "bedrooms", "minimum_nights"]
    data = pd.DataFrame([[price, bedrooms, minimum_nights, availability]], columns=fitur).astype(float)
    data[kolom_log] = np.log1p(data[kolom_log])
    cluster = kmeans.predict(scaler.transform(data))[0]

    st.success(f"Listing ini masuk ke **Cluster {cluster} ({profil.loc[cluster, 'nama']})**")
    st.write("Karakteristik cluster ini (median):")
    st.dataframe(profil.loc[[cluster]])

st.markdown("---")
st.subheader("Profil Setiap Cluster")
st.dataframe(profil)
st.bar_chart(profil.set_index("nama")["jumlah"])
