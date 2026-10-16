# 04_UTS_Praktikum_PI

# 🔍 Information Retrieval System

Sistem pencarian dokumen berbasis teks yang dibangun untuk memenuhi tugas mata kuliah *Information Retrieval. Program ini melakukan pengolahan teks (*text preprocessing), pengindeksan dokumen (indexing), serta penentuan relevansi dokumen menggunakan metode *Cosine Similarity*.

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

1. Menerapkan alur text preprocessing pada dokumen berbahasa Indonesia.
2. Membangun indeks pencarian yang efisien menggunakan pustaka *Whoosh*.
3. Mengukur dan mengurutkan relevansi dokumen terhadap kata kunci (query) pengguna menggunakan algoritma *Cosine Similarity*.

---

## ✨ Fitur Utama

- *Text Preprocessing*: Pembersihan simbol, case folding, tokenisasi, stopword removal, serta stemming kata berbahasa Indonesia.
- *Indexing Document*: Penyimpanan dan pengindeksan teks secara terstruktur agar pencarian berjalan cepat.
- *Query Search*: Pencarian kata kunci secara intuitif melalui input pengguna.
- *Similarity Ranking*: Pengurutan hasil pencarian berdasarkan skor kemiripan tertinggi menggunakan *Vector Space Model / Cosine Similarity*.
- *Dataset Limiter*: Pengaturan kuota dokumen yang diproses dari tiap sumber data untuk fleksibilitas performa.

---
