import time
import os
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from question1 import load_documents, load_queries, load_relevant, filtering_missing_docs
from question4 import preprocess_for_sklearn


BEST_TFIDF_PARAMETERS = {
    "ngram_range": (1, 1),
    "sublinear_tf": True,
    "min_df": 1,
    "max_df": 0.8,
    "norm": "l1",
}

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RESULTS_DIR = "results"
K_RANGE = range(2, 11)  # εύρος τιμών k προς εξέταση


def build_tfidf_representation(doc_ids, doc_texts, param=BEST_TFIDF_PARAMETERS):
    #Δημιουργεί την αναπαράσταση TF-IDF για τα έγγραφα
    print("\n Building TF-IDF representation...")
    start_time = time.time()
    vectorizer = TfidfVectorizer(**param, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(doc_texts)
    elapsed_time = time.time() - start_time
    print(f"TF-IDF representation built in {elapsed_time:.2f} seconds.")
    return tfidf_matrix.toarray(), elapsed_time


def build_embedding_representation(doc_texts, model_name=EMBEDDING_MODEL_NAME):
    #Δημιουργεί την αναπαράσταση embeddings για τα έγγραφα χρησιμοποιώντας ένα προεκπαιδευμένο μοντέλο SentenceTransformer
    from sentence_transformers import SentenceTransformer
 
    print(f"\nLoading pre-trained model '{model_name}'...")
    model = SentenceTransformer(model_name)
 
    print("Calculating embeddings for all documents...")
    start_time = time.time()
    embeddings = model.encode(
        doc_texts,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True,
    )
    elapsed_time = time.time() - start_time
    print(f"  Shape of embeddings matrix: {embeddings.shape}  (time: {elapsed_time:.4f}s)")
    return embeddings, elapsed_time

def evaluate_k_values(X, k_range=K_RANGE, label=""):
    silhouette_scores = []
    inertias = []
 
    print(f"\nΔοκιμή διαφορετικών τιμών k για {label}...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        sil = silhouette_score(X, labels, metric="cosine")
        silhouette_scores.append(sil)
        inertias.append(kmeans.inertia_)
        print(f"  k={k}: silhouette={sil:.4f}, inertia={kmeans.inertia_:.2f}")
 
    best_k = list(k_range)[int(np.argmax(silhouette_scores))]
    print(f"  -> Βέλτιστο k (max silhouette) για {label}: {best_k}")
    return best_k, silhouette_scores, inertias
 
 
def plot_k_selection(k_range, silhouette_scores, inertias, label, output_dir=RESULTS_DIR):
    #Δημιουργεί γραφήματα για την επιλογή του βέλτιστου αριθμού συστάδων (k) με βάση το Silhouette Score και την Inertia
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
    axes[0].plot(list(k_range), silhouette_scores, marker="o", color="steelblue")
    axes[0].set_xlabel("Αριθμός συστάδων (k)")
    axes[0].set_ylabel("Silhouette Score")
    axes[0].set_title(f"Silhouette Score ανά k - {label}")
    axes[0].grid(True, alpha=0.3)
 
    axes[1].plot(list(k_range), inertias, marker="o", color="coral")
    axes[1].set_xlabel("Αριθμός συστάδων (k)")
    axes[1].set_ylabel("Inertia")
    axes[1].set_title(f"Elbow Method - {label}")
    axes[1].grid(True, alpha=0.3)
 
    plt.tight_layout()
    fname = f"{output_dir}/k_selection_{label.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Αποθηκεύτηκε: {fname}")
 

def run_clustering(X, k, algorithm="kmeans"):
    if algorithm == "kmeans":
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
    elif algorithm == "agglomerative":
        model = AgglomerativeClustering(n_clusters=k)
    else:
        raise ValueError("Άγνωστος αλγόριθμος clustering")
    labels = model.fit_predict(X)
    return labels
 
 
def plot_pca_clusters(X, labels, title, output_path):
    #Δημιουργεί ένα γράφημα 2D με χρήση PCA για την οπτικοποίηση των συστάδων
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
 
    explained_var = pca.explained_variance_ratio_.sum() * 100
 
    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(
        X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=25, alpha=0.75, edgecolors="k", linewidths=0.3
    )
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(f"{title}\n(εξηγούμενη διακύμανση 2D: {explained_var:.1f}%)")
    plt.colorbar(scatter, label="Cluster")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Αποθηκεύτηκε: {output_path}")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
 
    print("Φόρτωση δεδομένων...")
    docs = load_documents("data/docs")
    relevant = load_relevant("data/relevant.txt")
    relevant = filtering_missing_docs(relevant, docs)
 
    doc_ids = list(docs.keys())
    doc_texts_raw = [docs[d] for d in doc_ids]     # για embeddings (φυσικό κείμενο)
    doc_texts_tfidf = [preprocess_for_sklearn(docs[d]) for d in doc_ids]  # για TF-IDF (ίδιο pipeline με Ερ4)
 
    print(f"Σύνολο εγγράφων: {len(doc_ids)}")
 
    # -------------------- TF-IDF αναπαράσταση --------------------
    tfidf_X, tfidf_time = build_tfidf_representation(doc_ids, doc_texts_tfidf)
 
    best_k_tfidf, sil_tfidf, inertia_tfidf = evaluate_k_values(tfidf_X, K_RANGE, label="TF-IDF")
    plot_k_selection(K_RANGE, sil_tfidf, inertia_tfidf, label="TF-IDF")
 
    labels_tfidf = run_clustering(tfidf_X, best_k_tfidf, algorithm="kmeans")
    plot_pca_clusters(
        tfidf_X, labels_tfidf,
        title=f"Clustering (KMeans, k={best_k_tfidf}) - Αναπαράσταση TF-IDF", output_path=f"{RESULTS_DIR}/clustering_tfidf_pca.png",
    )
 

    # -------------------- Embeddings αναπαράσταση --------------------
    try:
        emb_X, emb_time = build_embedding_representation(doc_texts_raw)
 
        best_k_emb, sil_emb, inertia_emb = evaluate_k_values(emb_X, K_RANGE, label="Embeddings")
        plot_k_selection(K_RANGE, sil_emb, inertia_emb, label="Embeddings")
 
        labels_emb = run_clustering(emb_X, best_k_emb, algorithm="kmeans")
        plot_pca_clusters(
            emb_X, labels_emb,
            title=f"Clustering (KMeans, k={best_k_emb}) - Αναπαράσταση Embeddings", output_path=f"{RESULTS_DIR}/clustering_embeddings_pca.png",
        )
 
        # -------------------- Σύγκριση TF-IDF vs Embeddings clustering --------------------
        ari = adjusted_rand_score(labels_tfidf, labels_emb) if best_k_tfidf == best_k_emb else None
 
        with open(f"{RESULTS_DIR}/results5.txt", "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("ΑΠΟΤΕΛΕΣΜΑΤΑ ΕΡΩΤΗΜΑΤΟΣ 5 - CLUSTERING\n")
            f.write("=" * 70 + "\n\n")
 
            f.write(f"TF-IDF -> βέλτιστο k: {best_k_tfidf} "
                    f"(silhouette={max(sil_tfidf):.4f}, χρόνος αναπαράστασης={tfidf_time:.4f}s)\n")
            f.write(f"Embeddings -> βέλτιστο k: {best_k_emb} "
                    f"(silhouette={max(sil_emb):.4f}, χρόνος αναπαράστασης={emb_time:.4f}s)\n\n")
 
            if ari is not None:
                f.write(f"Adjusted Rand Index (TF-IDF vs Embeddings, ίδιο k={best_k_tfidf}): {ari:.4f}\n")
                f.write("(τιμές κοντά στο 1 -> οι δύο ομαδοποιήσεις συμφωνούν, "
                        "κοντά στο 0 -> ουσιαστικά διαφορετικές ομαδοποιήσεις)\n\n")
            else:
                f.write("Τα δύο μοντέλα επέλεξαν διαφορετικό βέλτιστο k, "
                        "οπότε δεν υπολογίστηκε άμεση σύγκριση Adjusted Rand Index.\n"
                        "(Μπορείς να ξαναϋπολογίσεις clustering με κοινό k αν θέλεις άμεση σύγκριση.)\n\n")
 
            f.write("Παρατηρήσεις προς σχολιασμό στην αναφορά:\n")
            f.write("- Ποιο k επιλέχθηκε από κάθε αναπαράσταση και γιατί (elbow/silhouette).\n")
            f.write("- Αν οι συστάδες στο PCA scatter είναι συμπαγείς/διαχωρισμένες ή αλληλεπικαλύπτονται.\n")
            f.write("- Αν οι σημασιολογικές ενσωματώσεις (embeddings) δίνουν πιο διακριτές συστάδες\n"
                    "  σε σχέση με το αραιό TF-IDF, κάτι αναμενόμενο καθώς συλλαμβάνουν νοηματική\n"
                    "  ομοιότητα πέρα από την ακριβή επικάλυψη όρων.\n")
 
        print(f"\nΑποθηκεύτηκε στο: {RESULTS_DIR}/results5.txt")
 
    except ImportError:
        print("\nΗ βιβλιοθήκη sentence-transformers δεν είναι εγκατεστημένη.")
        print("Εγκατάστησε με: python -m pip install sentence-transformers")
        print("Το τμήμα TF-IDF ολοκληρώθηκε κανονικά, τα αποτελέσματα embeddings παραλείφθηκαν.")
 
    print("\nΟλοκληρώθηκε η ανάλυση του Ερωτήματος 5!")
 