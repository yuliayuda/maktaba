# maktaba

# Proyek Klasifikasi Teks Bahasa Arab

Proyek ini bertujuan untuk melatih model klasifikasi teks menggunakan dataset berbahasa Arab. Model ini menggunakan Transformers dari Hugging Face dan dapat digunakan untuk berbagai aplikasi pemrosesan bahasa alami.

## Struktur Proyek

- `data/`: Berisi dataset dan data yang sudah diproses.
- `models/`: Menyimpan model dan tokenizer yang telah dilatih.
- `notebooks/`: Notebook Jupyter untuk preprocessing, training, dan evaluation.
- `src/`: Skrip Python untuk preprocessing, training, evaluasi, dan inferensi.
- `requirements.txt`: Daftar pustaka yang diperlukan untuk menjalankan proyek.
- `README.md`: Deskripsi proyek dan panduan penggunaan.
- `config.yaml`: Pengaturan parameter pelatihan.

## Cara Menjalankan Proyek

1. **Persiapkan Lingkungan**:
   - Pastikan Anda memiliki Python 3.7 atau lebih baru.
   - Instal semua dependensi:
     ```bash
     pip install -r requirements.txt
     ```

2. **Preprocessing Data**:
   - Jalankan `preprocessing.ipynb` untuk membersihkan dan mempersiapkan dataset.

3. **Melatih Model**:
   - Jalankan `training.ipynb` untuk melatih model.

4. **Evaluasi Model**:
   - Jalankan `evaluation.ipynb` untuk mengevaluasi performa model.

## Kontribusi

Jika Anda ingin berkontribusi, silakan fork repository ini dan buat pull request.

## Struktur Folder

│
├── data/  
│    ├── dataset.csv              --> (Data teks utama berbahasa Arab)
│    └── preprocessed_data.csv    --> (Data yang sudah dibersihkan)
│
├── models/  
│    ├── arabic_sentence_transformer/  --> (Model yang sudah dilatih)
│    └── tokenizer/                --> (Tokenizer yang digunakan untuk model)
│
├── notebooks/  
│    ├── preprocessing.ipynb       --> (Notebook untuk membersihkan data)
│    ├── training.ipynb            --> (Notebook untuk melatih model)
│    └── evaluation.ipynb          --> (Notebook untuk mengevaluasi model)
│
├── src/  
│    ├── preprocess.py             --> (Skrip untuk membersihkan teks)
│    ├── train.py                  --> (Skrip utama untuk melatih model)
│    ├── evaluate.py               --> (Skrip untuk menguji model)
│    ├── infer.py                  --> (Skrip untuk membuat prediksi atau embedding)
│    └── save_load_model.py        --> (Skrip untuk menyimpan dan memuat model)
│
├── requirements.txt              --> (Daftar pustaka yang diperlukan)
├── README.md                     --> (Deskripsi proyek dan panduan penggunaan)
└── config.yaml                   --> (Pengaturan parameter pelatihan)

