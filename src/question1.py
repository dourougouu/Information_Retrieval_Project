import os
import re
from nltk.corpus import stopwords
from collections import defaultdict
relevant = defaultdict(dict)

# Διάβασμα και φόρτωση περιεχομένων των εγγράφων που υπάρχουν στον φάκελο data/docs/, queries.txt και relevant.txt
def load_documents(docs_path):
    docs={}
    for filename in os.listdir(docs_path):
        doc_id = os.path.splitext(filename)[0]
        file_path = os.path.join(docs_path, filename)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            text=file.read().strip()
        docs[doc_id]=text
    return docs

def load_queries(path):
    queries={}
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            qID, text = line.split("\t", 1)
            queries[qID] = text.strip()
    return queries

def load_relevant(path):
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            qID, docID, rel = line.split()
            relevant[qID][docID] = int(rel)
    return relevant


# Προεπεξεργασία κειμένου: ουσιαστικά, μετατρέπω κάθε κείμενο σε καθαρή μορφή-λίστα tokens
# Δηλαδή, μετατρέπω όλα τα γράμματα σε πεζά(lowercase), αφαιρώ σημεία στίξης, αριθμούς και 
# stopwords(the, is, at, a, and, κτλπ)

STOPWORDS = set(stopwords.words("english")) #Φόρτωση λίστας stopwords από την βιβλιοθήκη nltk
def preprocess_text(text):
    text=text.lower() #μετατροπή σε πεζά
    text=re.sub(r"[^a-z\s]", " ", text) #αφαίρεση σημείων στίξης και αριθμών (κρατάω μόνο πεζά γράμματα και κενά)
    tokens=text.split() #σπάσιμο κειμένου σε tokens με βάση τα κενά (tokenization)
    tokens= [t for t in tokens if t not in STOPWORDS and len(t)>2] #αφαίρεση stopwords και πολύ μικρών λέξεων (μήκους<=2)
    return tokens

#Δημιουργία του Αναστραμμένου Ευρετηρίου (Inverted Index)
def create_inverted_index(docs):
    inverted= defaultdict(dict)

    for docID, text in docs.items():
        tokens= preprocess_text(text)
        freqs= defaultdict(int)  #συχνότητες λέξεων/όρων
        for t in tokens:
            freqs[t] += 1
        for term, f in freqs.items():  #κάνω update το αναστραμμένο ευρετήριο
            inverted[term][docID] = f
    return inverted

if __name__ == "__main__":
    docs_path = "data/docs"

    docs = load_documents(docs_path)
    print("Σύνολο εγγράφων στον φάκελο docs:", len(docs))
    print(list(docs.keys())[:10])


    sample = list(docs.values())[0]
    print(preprocess_text(sample)[:40])


    inverted = create_inverted_index(docs)
    print("Σύνολο των όρων:", len(inverted))
    test = list(inverted.keys())[:5]
    for term in test:
        print(term, "->", list(inverted[term].items())[:3]) #εμφάνιση των πρώτων 3 εγγράφων που εμφανίζεται
                                                            #ο κάθε όρος 

