# 🔍 Information Retrieval System

Sistem pencarian dokumen berbasis teks yang dibangun untuk memenuhi tugas mata kuliah **Information Retrieval**. Program ini melakukan pengolahan teks (*text preprocessing*), pengindeksan dokumen (*indexing*), serta penentuan relevansi dokumen menggunakan metode **Cosine Similarity**.

## 📌 Daftar Isi

- [Latar Belakang & Tujuan](#-latar-belakang--tujuan)
- [Fitur Utama](#-fitur-utama)
- [Teknologi & Modul](#-teknologi--modul)
- [Sumber Dataset](#-sumber-dataset)
- [Struktur Proyek](#-struktur-proyek)
- [Panduan Instalasi & Penggunaan](#-panduan-instalasi--penggunaan)

---

## 🎯 Latar Belakang & Tujuan

Dalam era informasi modern, menemukan dokumen yang tepat di antara ribuan berkas merupakan tantangan tersendiri. Proyek ini bertujuan untuk:

1. Menerapkan alur *text preprocessing* pada dokumen berbahasa Indonesia.
2. Membangun indeks pencarian yang efisien menggunakan pustaka **Whoosh**.
3. Mengukur dan mengurutkan relevansi dokumen terhadap kata kunci (*query*) pengguna menggunakan algoritma **Cosine Similarity**.

---

## ✨ Fitur Utama

- **Text Preprocessing**: Pembersihan simbol, case folding, tokenisasi, stopword removal, serta stemming kata berbahasa Indonesia.
- **Indexing Document**: Penyimpanan dan pengindeksan teks secara terstruktur agar pencarian berjalan cepat.
- **Query Search**: Pencarian kata kunci secara intuitif melalui input pengguna.
- **Similarity Ranking**: Pengurutan hasil pencarian berdasarkan skor kemiripan tertinggi menggunakan *Vector Space Model* / *Cosine Similarity*.
- **Dataset Limiter**: Pengaturan kuota dokumen yang diproses dari tiap sumber data untuk fleksibilitas performa.

---

## 🛠️ Teknologi & Modul

Proyek ini dibangun menggunakan bahasa pemrograman **Python** beserta beberapa pustaka pendukung:

| Pustaka / Tools | Kegunaan |
| :--- | :--- |
| **Python** | Bahasa pemrograman utama |
| **Pandas** | Membaca dan mengolah file data CSV |
| **NumPy** | Operasi Vektor dan Array Numerik |
| **NLTK** | Tokenisasi teks |
| **Sastrawi** | Stemming kata dasar bahasa Indonesia |
| **Whoosh** | Mesin pencari dan indeksasi dokumen |
| **Scikit-Learn** | Pembuatan Bag of Words (BoW) & Cosine Similarity |

---

## 📂 Sumber Dataset

Program ini menggunakan 5 dataset berita dan karya ilmiah berbahasa Indonesia berbentuk file CSV:

1. `etd_usk.csv` — Repository ETD Universitas Syiah Kuala
2. `etd_ugm.csv` — Repository ETD Universitas Gadjah Mada
3. `kompas.csv` — Berita Portal Kompas
4. `tempo.csv` — Berita Portal Tempo
5. `mojok.csv` — Artikel Portal Mojok.co

> 💡 **Catatan:** Pastikan seluruh file CSV diletakkan di folder yang sesuai sebelum menjalankan program.

---

## 📁 Struktur Proyek

```text
.
├── data/
│   ├── etd_usk.csv
│   ├── etd_ugm.csv
│   ├── kompas.csv
│   ├── tempo.csv
│   └── mojok.csv
├── main.py
├── requirements.txt
└── README.md
```

---

## 🚀 Panduan Instalasi & Penggunaan

### 1. Clone Repository

```bash
git clone https://github.com/username/repository-name.git
cd repository-name
```

### 2. Install Dependensi

Pastikan modul yang diperlukan sudah terpasang:

```bash
pip install -r requirements.txt
```

Atau instal manual secara langsung:

```bash
pip install pandas numpy nltk sastrawi whoosh scikit-learn
```

### 3. Jalankan Program

```bash
python main.py
```

Masukkan *query* atau kata kunci yang ingin dicari saat program meminta input.

---

*Dibuat untuk memenuhi Tugas Mata Kuliah Information Retrieval.*
