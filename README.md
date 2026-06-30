# AthletiSense — Sport Injury Risk Prediction

Sistem prediksi risiko cedera atlet menggunakan **Artificial Neural Network (ANN)** berbasis web. Dikembangkan sebagai project mata kuliah Deep Learning, Program Studi Teknik Informatika, Politeknik Caltex Riau.

## Deskripsi

AthletiSense membantu pelatih, atlet, maupun tenaga medis untuk memprediksi tingkat risiko cedera atlet (rendah/tinggi) berdasarkan data biometrik dan kebiasaan latihan, lalu memberikan rekomendasi preskriptif yang dapat digunakan sebagai bahan pertimbangan dalam pengelolaan latihan.

## Fitur Utama

- Prediksi risiko cedera atlet secara real-time menggunakan model ANN (TensorFlow/Keras)
- Dashboard ringkasan statistik dataset dan faktor risiko utama
- Form input dengan perhitungan BMI otomatis
- Riwayat hasil prediksi tersimpan di database lokal (SQLite)
- Rekomendasi preskriptif berdasarkan tingkat probabilitas risiko (AMAN / WASPADA / BAHAYA)

## Dataset

Model dilatih menggunakan dataset **SIRP-600: Sports Injury Risk Prediction Dataset** dari Kaggle, berisi 600 data atlet dengan parameter biometrik, perilaku latihan, indikator fisiologis, dan riwayat cedera.

Sumber: [Kaggle - SIRP-600](https://www.kaggle.com/datasets/yuanchunhong/sirp-600-sports-injury-risk-prediction-dataset)

## Fitur Input Model (14 Fitur)

| Kategori | Fitur |
|---|---|
| Demografis & Fisik | Age, Gender, Height_cm, Weight_kg, BMI |
| Perilaku Latihan | Training_Frequency, Training_Duration, Warmup_Time, Training_Intensity |
| Fisiologis & Medis | Sleep_Hours, Flexibility_Score, Recovery_Time, Injury_History, Stress_Level |

> Catatan: fitur `Muscle_Asymmetry` yang sebelumnya digunakan pada versi awal model telah dikecualikan pada revisi terakhir, karena merupakan variabel persentase hasil pengukuran khusus yang kurang representatif untuk diisi secara mandiri oleh pengguna.

## Teknologi yang Digunakan

- **Backend**: Flask (Python)
- **Model**: TensorFlow / Keras (Artificial Neural Network)
- **Preprocessing**: scikit-learn (StandardScaler, LabelEncoder), imbalanced-learn (SMOTE)
- **Database**: SQLite
- **Frontend**: HTML, CSS, JavaScript (Vanilla)

## Struktur Project

```
SportInjuryWeb/
├── static/
├── templates/
│   ├── about.html
│   ├── dashboard.html
│   ├── predict.html
│   └── team.html
├── app.py
├── best_sport_injury_revisi_model.h5
├── scaler.pkl
├── predictions.db
└── requirements.txt
```

## Cara Menjalankan

### 1. Clone repository
```bash
git clone https://github.com/isyanpcr/SportInjuryRiskPrediction.git
cd SportInjuryRiskPrediction
```

### 2. Buat virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies
```bash
pip install flask tensorflow scikit-learn joblib numpy pandas
```

### 4. Jalankan aplikasi
```bash
python app.py
```

Buka browser ke `http://127.0.0.1:5000`

## Alur Prediksi

1. **Input Data Atlet** — pengguna mengisi 14 parameter biometrik dan latihan melalui form prediksi
2. **Preprocessing & Normalisasi** — encoding gender, standarisasi seluruh fitur numerik menggunakan StandardScaler
3. **Inferensi ANN** — model menghasilkan probabilitas risiko (0–1) melalui fungsi aktivasi Sigmoid
4. **Hasil & Rekomendasi** — threshold ≥ 0.5 dikategorikan sebagai risiko tinggi, sistem memberikan rekomendasi pencegahan

## Tim Pengembang

| Nama | NIM | Peran |
|---|---|---|
| Bimo Rahma Sakti | 2355301030 | Pembuatan Laporan |
| Muhammad Fajar Raffael | 2355301131 | Pembuatan Program Deep Learning |
| Muhamad Isyan Maulana | 2355301137 | Pembuatan Program Web |

**Dosen Pengampu & AIL:**
- Juni Nurma Sari, S.Kom., M.MT
- Ahmad Ali Munawar, S.Tr.Kom

## Lisensi

Project ini dibuat untuk keperluan akademik, Program Studi Teknik Informatika, Politeknik Caltex Riau, TA 2025/2026.
