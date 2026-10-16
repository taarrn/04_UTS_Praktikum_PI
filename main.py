import time
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from whoosh.index import open_dir, exists_in
from whoosh.qparser import QueryParser

from preprocessing import preprocess_text
from indexing import INDEX_DIR, load_and_index_datasets


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