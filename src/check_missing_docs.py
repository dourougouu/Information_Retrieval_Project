import os
from test import load_documents, load_relevant, filtering_missing_docs

docs = load_documents("data/docs/")
relevant = load_relevant("data/relevant.txt")
cleaned_relevant = filtering_missing_docs(relevant, docs)


# Βρίσκω ποια relevant docIDs λείπουν από τον φάκελο data/docs/
all_relevant_docs = set()
for qID, docs_info in relevant.items():
    all_relevant_docs.update(docs_info.keys())

all_relevant_docs_after_filtering = set()
for qID, docs_info in cleaned_relevant.items():
    all_relevant_docs_after_filtering.update(docs_info.keys())

missing_docs = sorted(all_relevant_docs - all_relevant_docs_after_filtering) 

with open("results/missing_docs.txt", "w", encoding="utf-8") as file:
    for d in missing_docs:
        file.write(d + "\n")


# Βρίσκω ποια docIDs λείπουν από τον φάκελο data/docs/
# Καθώς διαβάζουμε 1199 έγγραφα και τα IDs φτάνουν μέχρι 1239 (=> 40 missing doc files)
existing_doc_ids = set()

for d in docs:
    try:
        existing_doc_ids.add(int(d))
    except ValueError:
        pass

MAX_ID = 1239
all_possible_doc_ids = set(range(1, MAX_ID))

missing_numeric_doc_ids = sorted(all_possible_doc_ids - existing_doc_ids)

with open("results/missing_doc_IDs.txt", "w", encoding="utf-8") as file:
    for docID in missing_numeric_doc_ids:
        file.write(f"{docID}\n")
