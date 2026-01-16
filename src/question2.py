import os
import math
from collections import defaultdict
from question1 import (load_documents, load_queries, preprocess_text, create_inverted_index)


#Υπολογισμός του IDF για κάθε όρο στο ευρετήριο
def calculate_idf(inverted_index, N):
    idf = {}
    for term, postings in inverted_index.items():
        df = len(postings)
        idf[term] = math.log((N+1) / (df+1)) + 1
    return idf

#Υπολογισμός TF-IDF βάρων
def tfidf_weighting(tf, idf):
    return tf*idf

#Probabilistic inverse collection frequency factor
#p = log((N - n) / n , όπου N = συνολικός αριθμός εγγράφων, n = αριθμός εγγράφων με τον όρο
def calculate_paper_prob_idf(inverted_index, N):
    prob_idf = {}
    for term, postings in inverted_index.items():
        n = len(postings)
        #Προσθήκη smoothing για αποφυγή διαίρεσης με μηδέν ή αρνητικούς λογαρίθμους
        prob_idf[term] = math.log(max(1, (N - n + 0.5) / (n + 0.5)))
    return prob_idf

#n = 0.5 + 0.5 * (tf / max_tf)
def normalized_tf(tf, max_tf):
    if max_tf == 0:
        return 0
    return 0.5 + 0.5 * (tf/max_tf)


#Υπολογισμός λογαρίθμου TF-IDF βάρων
def log_tfidf(tf, idf):
    if tf>0:
      return (1 + math.log(tf))*idf
    return 0

#Δημιουργία διανύσματος εγγράφου
def doc_vector(docID, inverted_index, idf, scheme="tfidf", prob_idf=None):
    vector = {}

    #Για το probabilistic scheme χρειαζεται το max_tf για normalization
    if scheme == "probabilistic":
        max_tf=0
        for term, posting in inverted_index.items():
            if docID in posting:
                max_tf = max(max_tf, posting[docID])

    for term, posting in inverted_index.items():
        if docID in posting:
            tf = posting[docID]
            if scheme == "tfidf":
               vector[term] = tfidf_weighting(tf, idf[term])
            elif scheme == "log": 
               vector[term] = log_tfidf(tf, idf[term])
            elif scheme == "probabilistic":
                vector[term] = normalized_tf(tf, max_tf)
    
    if not vector:
        print(f"ΠΡΟΒΛΗΜΑ: Κένο doc vector για το doc: {docID[:50]}...")
    return vector

#Δημιουργία διανύσματος ερώτησης-query
def query_vector(query, idf, scheme="tfidf", prob_idf=None):
    tokens = preprocess_text(query)
    frequency = defaultdict(int)
    for t in tokens:
        frequency[t] += 1
    
    vector = {}
    for term, tf in frequency.items():
        if term in idf:
            if scheme == "tfidf":
                if term in idf:
                   vector[term] = tfidf_weighting(tf, idf[term])
            elif scheme == "log":
                if term in idf:
                   vector[term] = log_tfidf(tf, idf[term])
            elif scheme == "probabilistic":
                if prob_idf and term in prob_idf:
                    vector[term] = 1.0 * prob_idf[term]
    
    if not vector:
        print(f"ΠΡΟΒΛΗΜΑ: Κένο query vector για το query: {query[:50]}...")

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
def ranking_docs(query, docs, inverted_index, idf, scheme="tfidf", prob_idf=None):
    qr_vector = query_vector(query, idf, scheme, prob_idf)
    if not qr_vector:
        print(f"ΠΡΟΣΟΧΗ: Κενό query vector για: {query[:50]}...")
        return[]

    scores = []

    for docID in docs:
        d_vector = doc_vector(docID, inverted_index, idf, scheme, prob_idf)
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
    prob_idf = calculate_paper_prob_idf(inverted_index, len(docs))

    qID = list(queries.keys())[0]
    query = queries[qID]

    # Δοκιμή με όλα τα ερωτήματα (όχι μόνο το πρώτο)
    for qID, query in list(queries.items())[:3]:  # πρώτα 3 για αρχή
      print(f"\n{'='*50}")
      print(f"Query ID: {qID}")
      print(f"Query: {query[:80]}...")
    
      print("\nTF-IDF Top 3:")
      results_tfidf = ranking_docs(query, docs, inverted_index, idf, "tfidf")
      print(results_tfidf[:3])
    
      print("\nLOG TF-IDF Top 3:")
      results_log = ranking_docs(query, docs, inverted_index, idf, "log")
      print(results_log[:3])

      print("\nPROBABILISTIC (nxx*bpx) Top 3:")
      results_prob = ranking_docs(query, docs, inverted_index, idf, "probabilistic", prob_idf)
      for i, (docID, score) in enumerate(results_prob[:3], 1):
          print(f" {i}. {docID[:60]}... (score: {score:.4f})")