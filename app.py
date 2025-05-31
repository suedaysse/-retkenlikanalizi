import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import datetime

# 📦 Model ve yardımcı dosyaları yükle
model = joblib.load("ridge_model.pkl")
feature_means = joblib.load("ridge_model_means.pkl")
feature_columns = joblib.load("ridge_model_columns.pkl")

# 📁 Kalıcı kayıt dosyası
csv_path = "user_predictions.csv"
if not os.path.exists(csv_path):
    pd.DataFrame(columns=["Kullanıcı", "Tarih", "Tahmin", "Uyku Süresi", "Kafein Miktarı (mg)", "Ekran Süresi", "Egzersiz Süresi"]).to_csv(csv_path, index=False)

df = pd.read_csv(csv_path)

# Sayfa yapılandırması
st.set_page_config(page_title="Üretkenlik Tahmini", page_icon="💼")
st.title("💼 Üretkenlik Tahmini")

st.markdown("""
## 🧠 Ne Yapıyor?

Bu uygulama, uyku süresi, kafein alımı, ekran süresi ve egzersiz süresi gibi verileri kullanarak **günlük üretkenlik skorunuzu** tahmin eder.

🔸 Uygulama iki şekilde kullanılabilir:
- **Hızlı Tahmin Alanı:**  Verilerinizi girin, anında tahmini üretkenlik skorunuzu görün.
- **Takvim Paneli:**  Her gün için verilerinizi girin ve tahminleri saklayarak zaman içindeki değişimi takip edin.

📊 Tahminler, daha önce eğitilmiş bir **Makine Öğrenmesi Modeli (Ridge Regression)** ile yapılır. 
Bu sayede kendi verileriniz üzerinden günlük üretkenlik düzeyinizi görebilir ve alışkanlıklarınızla nasıl ilişkilendiğini anlayabilirsiniz.
""")

# ---------------------------------------
# 🔹 Öne Çıkan Tahmin Kutuları
# ---------------------------------------


# ---------------------------------------
# 🔍 Hızlı Tahmin Alanı
# ---------------------------------------
st.header("🔍 Hızlı Tahmin")
with st.form("quick_form"):
    sleep = st.slider("🛌 Uyku Süresi (saat)", 4.0, 10.0, 7.0, 0.1)
    caffeine = st.slider("☕ Kafein Miktarı (mg)", 0, 300, 150, 10)
    screen = st.slider("📱 Ekran Süresi (dk)", 0, 180, 90, 5)
    exercise = st.slider("🏃‍♀️ Egzersiz Süresi (dakika)", 0, 120, 30, 5)
    quick_submit = st.form_submit_button("📊 Tahmin Et")

    if quick_submit:
        user_input = feature_means.copy()
        user_input["Total Sleep Hours"] = sleep
        user_input["Caffeine Intake (mg)"] = caffeine
        user_input["Screen Time Before Bed (mins)"] = screen
        user_input["Exercise (mins/day)"] = exercise

        input_vector = np.array([user_input[col] for col in feature_columns]).reshape(1, -1)
        pred = model.predict(input_vector)[0]
        pred = np.clip(pred, 1, 10)

        st.success(f"✅ Tahmini Üretkenlik Skoru: **{round(pred, 2)} / 10**")
        st.progress(pred / 10)
# ---------------------------------------
# 📅 Takvim Paneli (Sidebar)
# ---------------------------------------
with st.sidebar.expander("📅 Günlük Tahmin Kaydı", expanded=False):
    name = st.text_input("👤 Adınız")
    date = st.date_input("📆 Tarih", value=datetime.date.today())

    with st.form("calendar_form"):
        sleep2 = st.slider("🛌 Uyku Süresi", 4.0, 10.0, 7.0, 0.1, key="sleep2")
        caffeine2 = st.slider("☕ Kafein Miktarı (mg)", 0, 300, 150, 10, key="caffeine2")
        screen2 = st.slider("📱 Ekran Süresi", 0, 180, 90, 5, key="screen2")
        exercise2 = st.slider("🏃‍♀️ Egzersiz Süresi (dk)", 0, 120, 30, 5, key="exercise2")
        save_submit = st.form_submit_button("💾 Kaydet")

        if save_submit and name.strip() != "":
            user_input = feature_means.copy()
            user_input["Total Sleep Hours"] = sleep2
            user_input["Caffeine Intake (mg)"] = caffeine2
            user_input["Screen Time Before Bed (mins)"] = screen2
            user_input["Exercise (mins/day)"] = exercise2

            input_vector = np.array([user_input[col] for col in feature_columns]).reshape(1, -1)
            prediction = model.predict(input_vector)[0]
            prediction = np.clip(prediction, 1, 10)

            new_data = pd.DataFrame([{
                "Kullanıcı": name,
                "Tarih": date,
                "Tahmin": round(prediction, 2),
                "Uyku Süresi": sleep2,
                "Kafein Miktarı (mg)": caffeine2,
                "Ekran Süresi": screen2,
                "Egzersiz Süresi": exercise2
            }])
            new_data.to_csv(csv_path, mode="a", header=False, index=False)

            st.success(f"📌 {name} için {date} günü tahmin kaydedildi.")
            st.progress(prediction / 10)

# ---------------------------------------
# 📂 Verileri Listele & Sil
# ---------------------------------------
st.markdown("---")
st.subheader("📂 Kayıtlı Veriler")
if not df.empty:
    user_list = df["Kullanıcı"].unique()
    selected_user = st.selectbox("👥 Kullanıcı Seçin", user_list)
    user_data = df[df["Kullanıcı"] == selected_user].sort_values(by="Tarih", ascending=False)
    st.dataframe(user_data.reset_index(drop=True))

    if st.button("🗑️ Bu Kullanıcının Verilerini Sil"):
        df = df[df["Kullanıcı"] != selected_user]
        df.to_csv(csv_path, index=False)
        st.success("Veriler silindi. Sayfayı yenileyin.")
else:
    st.info("Henüz kayıt yok.")
