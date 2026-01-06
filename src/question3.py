import io
from typing import List, Tuple
import matplotlib.pyplot as plt


def precision(retrieved: List[str], relevant: List[str]) ->float:
    if not retrieved:
        return 0.0
    
    retrieved_set = set(retrieved)
    relevant_set = set(relevant)

    true_positives = len(retrieved_set & relevant_set)

    return true_positives/len(retrieved)


def recall(retrieved: List[str], relevant: List[str]) -> float:
    if not relevant:
       return 0.0
    
    retrieved_set = set(retrieved)
    relevant_set = set(relevant)

    true_positives = len(retrieved_set & relevant_set)

    return true_positives/len(relevant_set)


def f1_score(retrieved: List[str], relevant: List[str]) -> float:
    p = precision(retrieved, relevant)
    r = recall(retrieved, relevant)

    if p+r == 0:
        return 0.0
    
    return 2*p*r/(p+r)


def precision_at_k(ranked_docs: List[str], relevant: List[str], k: int) -> float:
    if k<=0:
        return 0.0
    
    top_k=ranked_docs[:k]
    
    if not top_k:
        return 0.0
    
    relevant_set = set(relevant)
    true_positives = sum(1 for doc in top_k if doc in relevant_set)

    return true_positives/k


def precision_recall_curve(ranked_docs: List[str], relevant: List[str]) -> Tuple[List[float], List[float]]:
    if not relevant:
        return [0.0]*len(ranked_docs), [0.0]*len(ranked_docs)
    
    relevant_set = set(relevant)
    precisions = []
    recalls = []
    
    true_positives = 0
    total_relevant = len(relevant_set)

    for i, doc in enumerate(ranked_docs, start=1):
        if doc in relevant_set:
            true_positives +=1
        
        prc = true_positives/i
        rcl= true_positives/total_relevant
        

        precisions.append(prc)
        recalls.append(rcl)

    return recalls, precisions

def plot_precision_recall_curve(recalls: List[float], precisions: List[float]) -> None:
    """
    Plots Precision-Recall curve.
    """
    plt.figure()
    plt.plot(recalls, precisions, marker="o")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    ranked_docs = ["546", "321", "326", "592", "945"]
    relevant_docs = ["546", "326", "900"]

    print("Precision:", precision(ranked_docs, relevant_docs))
    print("Recall:", recall(ranked_docs, relevant_docs))
    print("F1-score:", f1_score(ranked_docs, relevant_docs))
    print("Precision@5:", precision_at_k(ranked_docs, relevant_docs, 5))

    recalls, precisions = precision_recall_curve(ranked_docs, relevant_docs)
    plot_precision_recall_curve(recalls, precisions)



