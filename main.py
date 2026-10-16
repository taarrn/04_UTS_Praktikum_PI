import os
import re
import time
import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from whoosh.index import create_in, open_dir, exists_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser


# Unduh resource NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Lokasi dataset dan indeks
DATASET_DIR = "dataset"
INDEX_DIR = "indexdir"

# Variabel stemmer dan stopword
stemmer = None
list_stopwords = set(stopwords.words('indonesian'))


def get_stemmer():
    """Memuat Sastrawi saat dibutuhkan."""
    global stemmer

    if stemmer is None:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()

    return stemmer


def preprocess_text(text, use_stemming=True):
    """Membersihkan dan memproses teks."""

    if not text or pd.isna(text):
        return ""

    # 1. Case folding
    text = str(text).lower()

    # 2. Menghapus tanda baca, angka, dan simbol
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # 3. Tokenisasi
    tokens = word_tokenize(text)

    # 4. Stopword removal dan menghapus kata pendek
    filtered = [
        token for token in tokens
        if token.isalpha()
        and token not in list_stopwords
        and len(token) > 2
    ]

    # 5. Stemming jika diaktifkan
    if use_stemming:
        active_stemmer = get_stemmer()
        filtered = [
            active_stemmer.stem(token)
            for token in filtered
        ]

    return " ".join(filtered)


def get_schema():
    """Menentukan struktur indeks Whoosh."""

    return Schema(
        id=ID(stored=True, unique=True),
        dataset=TEXT(stored=True),
        title=TEXT(stored=True),
        content=TEXT(stored=True),
        clean_content=TEXT(stored=True)
    )


def load_and_index_datasets(limit_per_file=None):
    """Membaca dataset dan melakukan indexing."""

    print("\n--- PROSES INDEXING DATASET ---")

    start_time = time.time()

    # Buat folder indeks jika belum ada
    os.makedirs(INDEX_DIR, exist_ok=True)

    schema = get_schema()
    ix = create_in(INDEX_DIR, schema)
    writer = ix.writer()

    total_docs = 0

    csv_files = [
        "etd_usk.csv",
        "etd_ugm.csv",
        "kompas.csv",
        "tempo.csv",
        "mojok.csv"
    ]

    try:
        for csv_file in csv_files:
            file_path = os.path.join(DATASET_DIR, csv_file)

            if not os.path.exists(file_path):
                print(f"File '{file_path}' tidak ditemukan, dilewati.")
                continue

            ds_name = csv_file.replace(".csv", "")

            print(
                f"Mengindeks file: {csv_file} "
                f"(Maksimal {limit_per_file} dokumen)..."
            )

            try:
                df = pd.read_csv(file_path)

                # Batasi jumlah dokumen per file jika limit diisi
                if limit_per_file is not None:
                    df = df.head(limit_per_file)

                for idx, row in df.iterrows():
                    title = str(
                        row.get('judul', f"Dokumen {idx + 1}")
                    )

                    content = str(row.get('konten', ''))

                    # Indexing tanpa stemming agar lebih ringan
                    cleaned_text = preprocess_text(
                        content,
                        use_stemming=False
                    )

                    doc_id = f"{ds_name}_{idx}"

                    writer.add_document(
                        id=doc_id,
                        dataset=ds_name,
                        title=title[:100],
                        content=content[:300],
                        clean_content=cleaned_text
                    )

                    total_docs += 1

            except Exception as e:
                print(f"Gagal membaca {file_path}: {e}")

        writer.commit()

    except Exception as e:
        writer.cancel()
        print(f"Indexing gagal: {e}")
        return

    elapsed = time.time() - start_time

    print("\n--- HASIL INDEXING ---")
    print(f"Limit per dataset : {limit_per_file}")
    print(f"Total dokumen     : {total_docs}")
    print(f"Waktu indexing    : {elapsed:.2f} detik\n")


def search_query(query_text, top_k=5):
    """Mencari dan mengurutkan dokumen berdasarkan cosine similarity."""

    if not exists_in(INDEX_DIR):
        print("\nIndeks belum ada. Jalankan menu [1] terlebih dahulu.\n")
        return

    start_time = time.time()

    # Preprocessing query tanpa stemming agar konsisten dengan indeks
    clean_query = preprocess_text(
        query_text,
        use_stemming=False
    )

    if not clean_query:
        print("\nQuery tidak valid atau hanya berisi stopword.\n")
        return

    ix = open_dir(INDEX_DIR)

    with ix.searcher() as searcher:
        parser = QueryParser(
            "clean_content",
            ix.schema
        )

        parsed_query = parser.parse(clean_query)
        results = searcher.search(parsed_query, limit=100)

        if len(results) == 0:
            print("\nTidak ada dokumen yang cocok dengan query tersebut.\n")
            return

        doc_list = []
        clean_docs = []

        for hit in results:
            doc_list.append({
                'id': hit['id'],
                'dataset': hit['dataset'],
                'title': hit['title'],
                'snippet': hit['content']
            })

            clean_docs.append(hit['clean_content'])

    # Representasi Bag of Words
    vectorizer = CountVectorizer()
    X_docs = vectorizer.fit_transform(clean_docs)
    query_vec = vectorizer.transform([clean_query])

    # Perhitungan cosine similarity
    scores = cosine_similarity(
        query_vec,
        X_docs
    ).flatten()

    ranked_indices = np.argsort(scores)[::-1][:top_k]

    elapsed = time.time() - start_time

    print(
        f"\nHasil Pencarian Top {top_k} Dokumen "
        f"({elapsed * 1000:.1f} ms):"
    )
    print("-" * 60)

    rank = 1

    for idx in ranked_indices:
        score = scores[idx]

        if score <= 0:
            continue

        doc = doc_list[idx]

        print(
            f"[{rank}] Cosine Similarity: {score:.4f} "
            f"| Dataset: {doc['dataset']}"
        )
        print(f"    Judul   : {doc['title']}")
        print(
            f"    Cuplikan: "
            f"{doc['snippet'].replace(chr(10), ' ')[:100]}..."
        )
        print("-" * 60)

        rank += 1

    if rank == 1:
        print("Tidak ada dokumen dengan skor kemiripan positif.")

    print()


def main():
    while True:
        print("=== INFORMATION RETRIEVAL SYSTEM ===")
        print("[1] Load & Index Dataset")
        print("[2] Search Query")
        print("[3] Exit")
        print("====================================")

        pilihan = input("Pilih menu: ").strip()

        if pilihan == "1":

            # Input limit dari pengguna
            batas = input(
                "\nMasukkan limit per dataset "
                "(Masukkan Rentang Dokumen): "
            ).strip()

            if batas.lower() == "none":
                limit = None

            else:
                try:
                    limit = int(batas)

                    if limit <= 0:
                        print("Limit harus lebih dari 0.\n")
                        continue

                except ValueError:
                    print("Masukkan angka yang valid atau None.\n")
                    continue

            load_and_index_datasets(
                limit_per_file=limit
            )

        elif pilihan == "2":
            query = input(
                "\nMasukkan query pencarian: "
            ).strip()

            if query:
                search_query(query, top_k=5)
            else:
                print("\nQuery tidak boleh kosong.\n")

        elif pilihan == "3":
            print("\nTerima kasih.")
            break

        else:
            print(
                f"\nPilihan '{pilihan}' tidak valid. "
                "Masukkan angka 1, 2, atau 3.\n"
            )


if __name__ == "__main__":
    main()