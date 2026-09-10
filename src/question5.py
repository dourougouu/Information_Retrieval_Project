import time
import os
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from question1 import load_documents
from question4 import preprocess_for_sklearn
from sentence_transformers import SentenceTransformer 


# Οι καλύτερες παράμετροι που βρέθηκαν στο Ερώτημα 4
BEST_TFIDF_PARAMETERS = {
    "ngram_range": (1, 1),
    "sublinear_tf": True,
    "min_df": 1,
    "max_df": 0.8,
    "norm": "l1",
}

#Pretrained μοντελο για Sentence Transformers embeddings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RESULTS_DIR = "results"
K_RANGE = range(2, 11)  # εύρος τιμών k προς εξέταση


def build_tfidf_representation(doc_texts):
    #Δημιουργεί την αναπαράσταση TF-IDF για τα έγγραφα 
    print("\n Building TF-IDF representation...")
    start_time = time.time()
    
    vectorizer = TfidfVectorizer(**BEST_TFIDF_PARAMETERS, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(doc_texts) #το αποτέλεσμα να μείνει sparse matrix

    elapsed_time = time.time() - start_time
    print(f"TF-IDF representation matrix shape: {tfidf_matrix.shape}, built in {elapsed_time:.2f} seconds.")
    return tfidf_matrix, vectorizer, elapsed_time


def build_embedding_representation(doc_texts):
    #Δημιουργεί την αναπαράσταση embeddings για τα έγγραφα χρησιμοποιώντας ένα προεκπαιδευμένο μοντέλο SentenceTransformer
 
    print(f"\nLoading pre-trained model embedding '{EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    start_time = time.time()

    embeddings = model.encode(
        doc_texts,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True,
    )

    elapsed_time = time.time() - start_time
    print(f"Shape of embeddings matrix: {embeddings.shape} with time: {elapsed_time:.4f}seconds.")
    return embeddings, elapsed_time

def evaluate_k_values(X, k_range=K_RANGE, representation_name=""):
    #Εδώ δοκιμάζουμε διαφορετικές τιμές του k υπολιγίζοντας το Silhouette Score και την Inertia
    # για κάθε τιμή k, ώστε να επιλέξουμε το βέλτιστο k (με το μεγαλύτερο Silhouette Score).
    print(f"Επιλογή k για: {representation_name}")
    print(f"*" * 50)

    silhouette_scores = []
    inertias = []
 
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        silhouette = silhouette_score(X, labels, metric="cosine") #Για τα δύο representations χρησιμοποιώ cosine similarity
        silhouette_scores.append(silhouette)
        inertias.append(kmeans.inertia_)
        
        print(f"k={k}: silhouette={silhouette:.4f}, inertia={kmeans.inertia_:.2f}")
 
    best_index = np.argmax(silhouette_scores)
    best_k = list(k_range)[best_index]
    print(f"Βέλτιστο k (max silhouette) για {representation_name}: {best_k}")
    return best_k, silhouette_scores, inertias
 
 
def plot_k_selection(k_range, silhouette_scores, inertias, representation_name, output_path):
    #Δημιουργεί γραφήματα για την επιλογή του βέλτιστου αριθμού συστάδων (k) με βάση το Silhouette Score 
    #και την Inertia/Elbow
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
 
    #~Silhouette~
    axes[0].plot(list(k_range), silhouette_scores, marker="o", color="steelblue")
    axes[0].set_xlabel("Αριθμός συστάδων (k)")
    axes[0].set_ylabel("Silhouette Score")
    axes[0].set_title(f"Silhouette Score ανά k - {representation_name}")
    axes[0].grid(True, alpha=0.3)
 
    #~Elbow~
    axes[1].plot(list(k_range), inertias, marker="o", color="coral")
    axes[1].set_xlabel("Αριθμός συστάδων (k)")
    axes[1].set_ylabel("Inertia")
    axes[1].set_title(f"Elbow Method - {representation_name}")
    axes[1].grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Αποθηκεύτηκε: {output_path}")
 

def run_clustering(X, k):
    #Εδώ εκτελεί το KMeans clustering για το επιλεγμενο κ
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    return labels, model
 
 
def plot_pca_clusters(X, labels, representation_name, output_path):
    #Δημιουργεί ένα γράφημα 2D με χρήση PCA για την οπτικοποίηση των συστάδων (scatter plots)
    
    print(f"\nCreating PCA visualization for {representation_name}...")
    if hasattr(X, "toarray"):
        X_for_the_PCA = X.toarray()  # Μετατροπή sparse matrix σε dense για PCA 
    else:
        X_for_the_PCA = X 
    
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_for_the_PCA)
 
    explained_variance = pca.explained_variance_ratio_.sum() * 100
 
    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1], 
        c=labels, 
        cmap="tab10", 
        s=25, alpha=0.75, edgecolors="k", linewidths=0.3
    )

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(f"{representation_name}\n(Explained Variance 2D: {explained_variance:.2f}%)")
    plt.colorbar(scatter,label="Cluster")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Αποθηκεύτηκε: {output_path}")
    plt.close()
    return explained_variance


def print_clustering_sizes(labels, representation_name):
    #Εδώ εμφανίζει πόσα documents υπάρχουν σε κάθε cluster.
    print(f"\nCluster sizes for {representation_name}:")

    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = {}

    for cluster, count in zip(unique, counts):
        cluster_sizes[int(cluster)] = int(count)
        print(f"Cluster {cluster}: {count} documents")
    return cluster_sizes


def get_top_terms_per_clst(tfidf_matrix, labels, vectorizer, top_n=10):
    #Εδώ εμφανίζει τα top-n terms για κάθε cluster με βάση το TF-IDF
    print(f"\nTop {top_n} terms per cluster:")
    feature_names = vectorizer.get_feature_names_out()
    top_terms = {}

    for cluster in sorted(np.unique(labels)):
    
        cluster_tfidf = tfidf_matrix[labels == cluster]
        mean_tfidf = np.asarray(cluster_tfidf.mean(axis=0)).ravel()
        top_indices = mean_tfidf.argsort()[-top_n:][::-1] #θεσεις σημαντικοτερων όρων
        terms = [feature_names[indx] for indx in top_indices]
        top_terms[int(cluster)] = terms
    return top_terms


def save_results(best_k_tfidf, sil_tfidf,inertia_tfidf,tfidf_cluster_sizes,tfidf_top_terms,tfidf_time,best_k_embeddings,
                 sil_embeddings, inertia_embeddings, embedding_cluster_sizes, embedding_time, pca_tfidf_variance, pca_embedding_variance, ari):
    #Αποθηκεύει τα αποτελέσματα σε ένα αρχείο results5.txt

    output_file = os.path.join(RESULTS_DIR, "results5.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("*" * 70 + "\n")
        f.write("ΑΠΟΤΕΛΕΣΜΑΤΑ ΕΡΩΤΗΜΑΤΟΣ 5 - CLUSTERING\n")
        f.write("*" * 70 + "\n\n")
        
        #
        f.write(f"1. TF-IDF Clustering-> best k: {best_k_tfidf} "
                f"(best silhouette={max(sil_tfidf):.4f}, time={tfidf_time:.4f}s)\n")

        f.write(f"\n Cluster sizes:\n")
        for cluster, size in tfidf_cluster_sizes.items():
            f.write(f"Cluster {cluster}: {size} documents\n")

        f.write(f"Top terms per cluster:\n")
        for cluster, terms in tfidf_top_terms.items():
            f.write(f"Cluster {cluster}: {', '.join(terms)}\n")

        f.write(f"\nPCA Explained Variance (TF-IDF): {pca_tfidf_variance:.2f}%\n\n")

        ##
        f.write(f"2. Embeddings -> best k: {best_k_embeddings} "
                f"(embedding model: {EMBEDDING_MODEL_NAME}), best silhouette={max(sil_embeddings):.4f}, time={embedding_time:.4f}s)\n")

        f.write(f"\n Cluster sizes:\n")
        for cluster, size in embedding_cluster_sizes.items():
            f.write(f"Cluster {cluster}: {size} documents\n")

        f.write(f"\nPCA Explained Variance (Embeddings): {pca_embedding_variance:.2f}%\n\n")

     
        ###
        f.write(f"3.Comparison TF-IDF vs Embeddings clustering:\n")
        f.write(f"TF-IDF best k: {best_k_tfidf}\n")
        f.write(f"Embeddings best k: {best_k_embeddings}\n")

        f.write(f"TF-IDF best silhouette: {max(sil_tfidf):.4f}\n")
        f.write(f"Embeddings best silhouette: {max(sil_embeddings):.4f}\n")
        f.write(f"Adjusted Rand Index: {ari:.4f}\n")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
 
    print("Φόρτωση δεδομένων...")
    docs = load_documents("data/docs") 
    doc_ids = list(docs.keys())

    doc_texts_raw = [docs[doc_id] for doc_id in doc_ids]     # για embeddings (φυσικό κείμενο)
    doc_texts_tfidf = [preprocess_for_sklearn(docs[doc_id]) for doc_id in doc_ids]  # για TF-IDF (ίδιο pipeline με Ερ4)
 
    print(f"Σύνολο εγγράφων: {len(doc_ids)}")
 
    # -------------------- TF-IDF αναπαράσταση --------------------
    tfidf_X, vectorizer, tfidf_time = (build_tfidf_representation(doc_texts_tfidf))
 
    best_k_tfidf, sil_tfidf, inertia_tfidf = evaluate_k_values(tfidf_X, representation_name="TF-IDF")
    plot_k_selection(K_RANGE, sil_tfidf, inertia_tfidf, "TF-IDF", os.path.join(RESULTS_DIR, "k_selection_tfidf.png"))
 
    #clustering
    labels_tfidf, tfidf_model = run_clustering(tfidf_X, best_k_tfidf)
    
    tfidf_cluster_sizes = print_clustering_sizes(labels_tfidf, "TF-IDF")

    tfidf_top_terms = get_top_terms_per_clst(tfidf_X, labels_tfidf, vectorizer, top_n=10)
    print("\nTop terms per TF-IDF cluster:")

    for cluster, terms in tfidf_top_terms.items():
        print(f"Cluster {cluster}: {', '.join(terms)}")

    #pca
    pca_tfidf_variance = plot_pca_clusters(tfidf_X, labels_tfidf, "TF-IDF", os.path.join(RESULTS_DIR, "pca_tfidf.png"))

    #embeddings representation
    try:

        emb_X, embedding_time = (build_embedding_representation(doc_texts_raw))

        # Επιλογή k
        best_k_embeddings, sil_embeddings, inertia_embeddings = evaluate_k_values(emb_X,representation_name="Embeddings")

        # Γράφημα k selection
        plot_k_selection(K_RANGE, sil_embeddings, inertia_embeddings, "Embeddings", os.path.join(RESULTS_DIR, "embeddings_k_selection.png"))

        # Clustering
        labels_embeddings, embedding_model = (run_clustering(emb_X, best_k_embeddings))

        # Μέγεθος clusters
        embedding_cluster_sizes = (print_clustering_sizes(labels_embeddings, "Embeddings"))

        # PCA
        pca_embedding_variance = plot_pca_clusters(emb_X, labels_embeddings, "Embeddings", os.path.join(RESULTS_DIR, "embeddings_pca.png"))


        # Σύγκριση των δύο clusterings
        # Το ARI συγκρίνει τις δύο ομαδοποιήσεις των ίδιων documents. Δεν απαιτεί ίδιο αριθμό clusters.
        ari = adjusted_rand_score(labels_tfidf, labels_embeddings)
        print("COMPARISON OF TF-IDF AND EMBEDDINGS")

        print(f"TF-IDF best k: {best_k_tfidf}")
        print(f"Embeddings best k: {best_k_embeddings}")
        print(f"TF-IDF silhouette: {max(sil_tfidf):.4f}")
        print(f"Embeddings silhouette: {max(sil_embeddings):.4f}")
        print(f"Adjusted Rand Index: {ari:.4f}")


        # Αποθήκευση
        save_results(
            best_k_tfidf,
            sil_tfidf,
            inertia_tfidf,
            tfidf_cluster_sizes,
            tfidf_top_terms,
            tfidf_time,
            best_k_embeddings,
            sil_embeddings,
            inertia_embeddings,
            embedding_cluster_sizes,
            embedding_time,
            pca_tfidf_variance,
            pca_embedding_variance,
            ari
        )

    except ImportError:
        print("\nDidnt find sentence-transformers.")
