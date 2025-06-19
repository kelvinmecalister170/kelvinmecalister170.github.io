import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# Konfigurasi awal
st.set_page_config(page_title="Data Mining CSV App", layout="wide")
st.markdown("<h1 style='text-align: center;'>📘 Aplikasi Data Mining: Klasifikasi & Regresi</h1>", unsafe_allow_html=True)
st.markdown("---")

# Upload File
with st.sidebar:
    st.header("📂 Upload Dataset")
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

# Main logic
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("👁️‍🗨️ Preview Data")
    st.dataframe(df.head())

    st.sidebar.header("⚙️ Konfigurasi Kolom")
    all_columns = df.columns.tolist()

    # Pilih target dan fitur
    target = st.sidebar.selectbox("🎯 Pilih Target", all_columns)
    fitur = st.sidebar.multiselect("🧩 Pilih Fitur", [c for c in all_columns if c != target], default=[c for c in all_columns if c != target][:5])

    # Pilih algoritma
    algoritma = st.sidebar.selectbox("📌 Pilih Algoritma", ["Regresi Logistik", "Regresi Linier"])

    if fitur and target:
        df_model = df[fitur + [target]].dropna()
        X = df_model[fitur]
        y = df_model[target]

        # Encode fitur (jika string)
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        # Encode target jika perlu
        label_info = None
        if y.dtype == 'object':
            le_y = LabelEncoder()
            y = le_y.fit_transform(y.astype(str))
            label_info = le_y.classes_
        else:
            label_info = np.unique(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # ==============================
        # LOGISTIC REGRESSION
        # ==============================
        if algoritma == "Regresi Logistik":
            unique_classes = np.unique(y)
            if len(unique_classes) != 2:
                st.error(f"❌ Regresi Logistik hanya bisa untuk klasifikasi biner (2 kelas). Ditemukan {len(unique_classes)} kelas: {unique_classes}")
            else:
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)

                st.markdown("### ✅ Hasil Klasifikasi")
                st.write(f"**Akurasi Model:** {acc:.2%}")

                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
                            xticklabels=label_info, yticklabels=label_info, ax=ax)
                ax.set_xlabel("Prediksi")
                ax.set_ylabel("Aktual")
                ax.set_title("Confusion Matrix")
                st.pyplot(fig)

        # ==============================
        # LINEAR REGRESSION
        # ==============================
        elif algoritma == "Regresi Linier":
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            st.markdown("### 📈 Hasil Regresi")
            st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")

            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.set_xlabel("Aktual")
            ax.set_ylabel("Prediksi")
            ax.set_title("Grafik Prediksi vs Aktual")
            st.pyplot(fig)

        st.success("✅ Model berhasil dijalankan.")
else:
    st.info("Silakan upload file CSV untuk memulai.")
