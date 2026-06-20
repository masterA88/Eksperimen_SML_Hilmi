# Eksperimen_SML_Hilmi

Eksperimen dan otomatisasi preprocessing untuk submission **Membangun Sistem
Machine Learning (MSML)** — Kriteria 1.

Dataset: **Telco Customer Churn** (klasifikasi biner `Churn`).

## Struktur

```
Eksperimen_SML_Hilmi/
├── .github/workflows/preprocessing.yml   # GitHub Actions: preprocessing otomatis (Advanced)
├── telco_raw.csv                         # dataset mentah
└── preprocessing/
    ├── Eksperimen_Hilmi.ipynb            # eksplorasi manual: loading, EDA, preprocessing
    ├── automate_Hilmi.py                 # konversi eksperimen -> fungsi preprocessing otomatis
    └── telco_preprocessing/              # output siap latih
        ├── train.csv
        ├── test.csv
        └── scaler.pkl
```

## Menjalankan preprocessing otomatis

```bash
python preprocessing/automate_Hilmi.py \
  --input telco_raw.csv \
  --output preprocessing/telco_preprocessing
```

Fungsi `preprocess_data()` mengembalikan `X_train, X_test, y_train, y_test,
scaler` yang siap digunakan untuk pelatihan model pada kriteria berikutnya.

## Workflow CI

`.github/workflows/preprocessing.yml` menjalankan ulang preprocessing setiap kali
ada perubahan pada dataset mentah atau skrip otomatisasi (atau via
*workflow_dispatch*), lalu mengunggah dataset hasil proses sebagai artifact yang
dapat diunduh (dependensi di-pin agar hasilnya reproducible).

---
*Author: Hilmi — https://master-hilmi.vercel.app/*
