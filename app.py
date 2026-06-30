import logging
from pathlib import Path

import sqlite3
from datetime import datetime

from flask import Flask, jsonify, render_template, request
import joblib
import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'best_sport_injury_revisi_model.h5'
SCALER_PATH = BASE_DIR / 'scaler.pkl'

try:
    logger.info('Memuat model ANN dan scaler...')
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info('Model dan scaler berhasil dimuat.')
except Exception as exc:
    logger.exception('Gagal memuat model atau scaler: %s', exc)
    model = None
    scaler = None
 
 
# ── ROUTES ──────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('dashboard.html')


@app.route('/predict-page')
def predict_page():
    return render_template('predict.html')
 
 
@app.route('/about')
def about():
    return render_template('about.html')
 
@app.route('/team')
def team():
    return render_template('team.html')
 
@app.route('/get-history')
def get_history():
    with get_db() as conn:
        rows = conn.execute(
            'SELECT * FROM prediction_history ORDER BY id DESC LIMIT 50'
        ).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route('/delete-history/<int:record_id>', methods=['DELETE'])
def delete_history(record_id):
    with get_db() as conn:
        conn.execute('DELETE FROM prediction_history WHERE id = ?', (record_id,))
        conn.commit()
    return jsonify({'success': True})

# ── PREDICT API ──────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model atau scaler tidak ditemukan.'}), 500

    try:
        fields = [
            'Age', 'Gender', 'Height_cm', 'Weight_kg', 'BMI',
            'Training_Frequency', 'Training_Duration', 'Warmup_Time',
            'Sleep_Hours', 'Flexibility_Score',
            'Recovery_Time', 'Injury_History', 'Stress_Level',
            'Training_Intensity'
        ]

        values = [float(request.form.get(field, 0)) for field in fields]
        input_data = np.array([values])

        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data, verbose=0)

        probability = float(prediction[0][0])
        prob_percent = probability * 100

        if probability >= 0.5:
            risk_status = 'Risiko Cedera Tinggi'
            recommendation = (
                'Metrik atlet menunjukkan indikasi risiko cedera yang tinggi. '
                'Turunkan intensitas latihan, tambah durasi recovery dan pemanasan, '
                'serta jadwalkan observasi biomekanik dengan fisioterapis.'
            )
        else:
            risk_status = 'Risiko Cedera Rendah'
            recommendation = (
                'Parameter latihan atlet berada dalam zona risiko cedera yang rendah. '
                'Pertahankan konsistensi pola istirahat, intensitas latihan yang terukur, '
                'dan rutinitas pemanasan untuk menjaga performa.'
            )

        gender_label = 'Pria' if values[1] == 1 else 'Wanita'
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with get_db() as conn:
            conn.execute('''
                INSERT INTO prediction_history
                    (timestamp, age, gender, bmi, risk_status, prob_percent, recommendation)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, values[0], gender_label, values[4],
                risk_status, round(prob_percent, 2), recommendation))
            conn.commit()

        return jsonify({
            'risk_status': risk_status,
            'prob_percent': prob_percent,
            'recommendation': recommendation,
        })
    except ValueError as exc:
        logger.exception('Input tidak valid: %s', exc)
        return jsonify({'error': 'Input tidak valid. Pastikan format angka benar.'}), 400
    except Exception as exc:
        logger.exception('Kesalahan saat memproses prediksi: %s', exc)
        return jsonify({'error': 'Terjadi kesalahan server.'}), 500
 
 
# ── DATABASE SETUP ────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect('predictions.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                age REAL,
                gender TEXT,
                bmi REAL,
                risk_status TEXT,
                prob_percent REAL,
                recommendation TEXT
            )
        ''')
        conn.commit()

# buat tabel saat aplikasi start
init_db()

if __name__ == '__main__':
    app.run(debug=True, port=5000)