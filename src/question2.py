import os
import math
from collections import defaultdict
from question1 import (load_documents, load_queries, preprocess_text, create_inverted_index)


#Υπολογισμός του IDF για κάθε όρο στο ευρετήριο
def calculate_idf(inverted_index, N):
    idf = {}
    for term, postings in inverted_index.items():
        idf[term] = math.log(N / len(postings))
    return idf

#Υπολογισμός TF-IDF βαρών για κάθε έγγραφο και ερώτημα 
def tfidf_weighting(tf, idf):
    return tf*idf

#Υπολογισμός λογαρίθμου TF-IDF βαρών για κάθε έγγραφο και ερώτημα 
def log_tfidf(tf, idf):
    return (1 + math.log(tf))*idf

#Δημιουργία διανύσματος εγγράφου
def doc_vector(docID, inverted_index, idf, scheme="tfidf"):
    vector = {}
    for term, posting in inverted_index.items():
        if docID in posting:
            tf = posting[docID]
            vector[term] = tfidf_weighting(tf, idf[term]) if scheme == "tfidf" \
                else log_tfidf(tf, idf[term])
    return vector

#Δημιουργία διανύσματος ερώτησης-query
def query_vector(query, idf, scheme="tfidf"):
    tokens = preprocess_text(query)
    frequency = defaultdict(int)
    for t in tokens:
        frequency[t] += 1
    
    vector = {}
    for term, tf in frequency.items():
        if term in idf:
            vector[term] = tfidf_weighting(tf, idf[term]) if scheme == "tfidf" \
                else log_tfidf(tf, idf[term])
    return vector

#Υπολιγισμός συσχέτισης με εφαρμογή συνημίτονου της γωνίας για 
#εύρεση ομοιότητας μεταξύ του ερωτήματος και των εγγράφων
def cos_similarity(vector1, vector2):
    dot = sum( vector1[t] * vector2[t] for t in vector1 if t in vector2 )
    norm1 = math.sqrt(sum( v*v for v in  vector1.values() ))
    norm2 = math.sqrt(sum( v*v for v in  vector2.values() ))
    return 0 if norm1==0 or norm2==0 else dot / (norm1*norm2)

#Ταξινόμηση εγγράφων με βάση τη βαθμολογία ομοιότητας που υπολογίστηκε απο το cosine
#και επιστροφή των πιο σχετικών εγγράφων
def ranking_docs(query, docs, inverted_index, idf, scheme="tfidf"):
    qr_vector = query_vector(query, idf, scheme)
    scores = []

    for docID in docs:
        d_vector = doc_vector(docID, inverted_index, idf, scheme)
        score = cos_similarity(qr_vector, d_vector)
        if score>0:
           scores.append((docID, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
        
if __name__ == "__main__":
    docs = load_documents("data/docs")
    queries = load_queries("data/queries.txt")

    inverted_index = create_inverted_index(docs)
    idf = calculate_idf(inverted_index, len(docs))

    qID = list(queries.keys())[0]
    query = queries[qID]

    print("Αποτελέσματα TF-IDF\n")
    results = ranking_docs(query, docs, inverted_index, idf, "tfidf")
    print(results[:5])  #εμφάνιση των 5 πιο σχετικών εγγράφων

    print("Αποτελέσματα LOG TF-IDF\n")
    results = ranking_docs(query, docs, inverted_index , idf, "log")
    print(results[:5]) 

    