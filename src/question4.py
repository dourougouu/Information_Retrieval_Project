import time
import os
import numpy as np 
import itertools
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from question1 import load_documents, load_queries, load_relevant, filtering_missing_docs
from question2 import ranking_docs, calculate_idf, create_inverted_index, calculate_paper_prob_idf
from question3 import precision, recall, f1_score, precision_at_k

def evaluate_model(ranked_results, relevant_docs, k_values=[5, 10, 20]):
    #αξιολογεί τα αποτελέσματα της ανάκτησης με βάση τις μετρικές του ερωτήματος 3
    #k_values: λίστα με τις τιμές k για τον υπολογισμό της precision@k

    retrieved = [doc_ID for doc_ID, score in ranked_results]
    metrics = {
        'precision': precision(retrieved, relevant_docs),
        'recall': recall(retrieved, relevant_docs),
        'f1_score': f1_score(retrieved, relevant_docs)
    }
    for k in k_values:
        metrics[f'precision@{k}'] = precision_at_k(retrieved, relevant_docs, k)
    
    return metrics

def preprocess_for_sklearn(text):
    #Απλή προεπεξεργασία κειμένου για χρήση με sklearn. 
    #Το TfidfVectorizer κάνει μόνο του την δική του προεπεξεργασία 
    return text.lower()
    
def sklearn_tfidf_retrieval(queries, docs, vector, k_values=[5, 10, 20]):
    #Ανάκτηση με χρήση του TfidfVectorizer από την sklearn
    #Επιστρέφει τα αποτελέσματα της ανάκτησης (results)
    #Τον χρόνο δεικτοδότησης (fit_time) και τον συνολικό χρόνο ερωτημάτων (query_time)

    #Προεπεξεργασία εγγράφων
    doc_IDs = list(docs.keys())
    doc_texts = [preprocess_for_sklearn(docs[doc_ID]) for doc_ID in doc_IDs]

    #Δεικτοδότηση με TfidfVectorizer
    start_time = time.time()
    doc_vectors = vector.fit_transform(doc_texts)
    fit_time = time.time() - start_time

    #Ανάκτηση για κάθε query
    results = {}
    start_time = time.time()
        
    for qID, query_text in queries.items():
        query_vector = vector.transform([preprocess_for_sklearn(query_text)])
        similarities = cosine_similarity(query_vector, doc_vectors)[0]
        ranked = [(doc_IDs[i], similarities[i]) for i in range(len(doc_IDs)) if similarities[i]>0] #δημιουργία λίστας (docID, score) και ταξινόμηση
        ranked.sort(key=lambda x: x[1], reverse=True)
        results[qID] = ranked

    query_time = time.time() - start_time
    return results, fit_time, query_time

def test_tfidf_parameters(queries, docs, relevant, k_values=[5, 10, 20]):
    #Δοκιμάζει διαφορετικούς συνδυασμούς παραμέτρων του TfidfVectorizer και αξιολογεί τα αποτελέσματα

    #Ορισμός παραμέτρων για την δομική
    parameter_grid={
        'ngram_range': [(1,1), (1, 2), (1, 3)],
        'sublinear_tf': [True, False],
        'min_df': [1, 2, 3],
        'max_df': [0.8, 0.9, 1.0],
        'norm': ['l1', 'l2', None]
    }

    #Δημιουργία συνδυασμών
    parameter_combinations=[]
    for ngram in parameter_grid['ngram_range']:
        for sublinear in parameter_grid['sublinear_tf']:
            for min_df in parameter_grid['min_df']:
                for max_df in parameter_grid['max_df']:
                    for norm in parameter_grid['norm']:
                        parameter_combinations.append( {
                            'ngram_range': ngram,
                            'sublinear_tf': sublinear,
                            'min_df': min_df,
                            'max_df': max_df,
                            'norm': norm
                        })
    print(f"Συνολικός αριθμός συνδυασμών: {len(parameter_combinations)}")
    results_list=[]  #λίστα για αποθήκευση αποτελεσμάτων

    for idx, param in enumerate(parameter_combinations, 1):
        print(f"\nΔοκιμή {idx}/{len(parameter_combinations)}: {param}")
        vector = TfidfVectorizer(**param, stop_words='english')
        try:
            ranked_results, fit_time, query_time = sklearn_tfidf_retrieval(queries, docs, vector, k_values)
            #Υπολογισμός μέσων μετρικών για όλα τα ερωτήματα
            all_metrics=[]
            for qID in queries.keys():
                if qID in relevant and qID in ranked_results:
                    relevant_docs = list(relevant[qID].keys())
                    metrics = evaluate_model(ranked_results[qID], relevant_docs, k_values)
                    all_metrics.append(metrics)
                    
            #Υπολογισμός μέσων όρων
            avg_metrics={
                'precision': np.mean([m['precision'] for m in all_metrics]),
                'recall': np.mean([m['recall'] for m in all_metrics]),
                'f1_score': np.mean([m['f1_score'] for m in all_metrics])
            }
            for k in k_values:
                avg_metrics[f'precision@{k}'] = np.mean([m[f'precision@{k}'] for m in all_metrics])

            #αποθήκευση αποτελεσμάτων 
            results={
                'param': str(param),
                'fit_time': fit_time,
                'query_time': query_time,
                **avg_metrics
            }
            results_list.append(results)

            print(f" F1-Score: {avg_metrics['f1_score']:.4f}, Precision@10: {avg_metrics['precision@10']:.4f}")

        except Exception as e:
            print(f"ΣΦΑΛΜΑ: {e}")
            continue

    best_idx = max(range(len(results_list)), key=lambda i:results_list[i]['f1_score'])
    best_result = results_list[best_idx]
    best_param = parameter_combinations[best_idx]

    print("\n" + "="*70)
    print("ΒΕΛΤΙΣΤΕΣ ΠΑΡΑΜΕΤΡΟΙ:")
    print(best_param)
    print(f"\nΑποτελέσματα:")
    print(f"   Precision: {best_result['precision']:.4f}")
    print(f"   Recall: {best_result['recall']:.4f}")
    print(f"   F1-Score: {best_result['f1_score']:.4f}")
    print(f"   Precision@10: {best_result['precision@10']:.4f}")
    print("="*70)
    return results_list, best_param

def compare_all_models_used(queries, docs, relevant, inverted_index, idf, prob_idf, best_param, k_values=[5, 10,20]):
    print("\n" + "="*70)
    print("ΣΥΓΚΡΙΣΗ ΟΛΩΝ ΤΩΝ ΜΟΝΤΕΛΩΝ")
    print("="*70)
    models_results={}

    #1 TF-IDF από ερώτημα 2
    print("\n1. Υπολογισμός για TF-IDF (Ερώτημα 2)...")
    start_time = time.time()
    tfidf_results = {}
    for qID, query_text in queries.items():
        ranked = ranking_docs(query_text, docs, inverted_index, idf, "tfidf")
        tfidf_results[qID] = ranked
    tfidf_time = time.time() - start_time
    models_results['TF-IDF'] = (tfidf_results, tfidf_time)

    #2 Log TF-IDF από ερώτημα 2
    print("2. Υπολογισμός για Log TF-IDF (Ερώτημα 2)...")
    start_time = time.time()
    log_results = {}
    for qID, query_text in queries.items():
        ranked = ranking_docs(query_text, docs, inverted_index, idf, "log")
        log_results[qID] = ranked
    log_time = time.time() - start_time
    models_results['Log TF-IDF'] = (log_results, log_time)

    #3 Probabilistic από ερώτημα 2
    print("3. Υπολογισμός για Probabilistic (Ερώτημα 2)...")
    start_time = time.time()
    prob_results = {}
    for qID, query_text in queries.items():
        ranked = ranking_docs(query_text, docs, inverted_index, idf, "probabilistic", prob_idf)
        prob_results[qID] = ranked
    prob_time = time.time() - start_time
    models_results['Probabilistic'] = (prob_results, prob_time)

    #4 Sklearn TF-IDF με βέλτιστες παραμέτρους
    print("4. Υπολογισμός για Sklearn TF-IDF (βέλτιστες παράμετροι)")
    vector = TfidfVectorizer(**best_param, stop_words='english')
    sklearn_results, sklearn_fit_time, sklearn_query_time = sklearn_tfidf_retrieval(queries, docs, vector, k_values)
    sklearn_total_time = sklearn_fit_time + sklearn_query_time
    models_results['Sklearn TF-IDF'] = (sklearn_results, sklearn_total_time)

    #Υπολογισμός μετρικών για κάθε μονλέλο
    print("\n" + "-"*70)
    print(f"{'Μοντέλο':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Χρόνος (sec)':<12}")
    print("-"*70)

    comparison_data=[]

    for model_name, (results, exec_time) in models_results.items():
        all_metrics=[]
        for qID in queries.keys():
            if qID in relevant and qID in results:
                relevant_docs = list(relevant[qID].keys())
                metrics = evaluate_model(results[qID], relevant_docs, k_values)
                all_metrics.append(metrics)
        avg_precision = np.mean([m['precision'] for m in all_metrics])
        avg_recall = np.mean([m['recall'] for m in all_metrics])
        avg_f1 = np.mean([m['f1_score'] for m in all_metrics])
        avg_precisionat10 = np.mean([m['precision@10'] for m in all_metrics])
        print(f"{model_name:<20} {avg_precision:<12.4f} {avg_recall:<12.4f} {avg_f1:<12.4f} {avg_precisionat10:<12.4f} {exec_time:<12.4f}")

        comparison_data.append({
            'model': model_name,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': avg_f1,
            'precision@10': avg_precisionat10,
            'execution_time': exec_time
        })
    print("-"*70)
    return comparison_data

def visualization_of_data(comparison_data, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    
    models = [d['model'] for d in comparison_data]
    precisions = [d['precision'] for d in comparison_data]
    recalls = [d['recall'] for d in comparison_data]
    f1_scores = [d['f1_score'] for d in comparison_data]
    precisions_at_10 = [d['precision@10'] for d in comparison_data]
    times = [d['execution_time'] for d in comparison_data]

    #γραγηματα
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['steelblue', 'coral', 'seagreen', 'purple']
    
    for i, (model, prec, rec) in enumerate(zip(models, precisions, recalls)):
        ax.scatter(rec, prec, s=300, alpha=0.6, color=colors[i], label=model, edgecolors='black', linewidths=1.5)
        ax.annotate(model, (rec, prec), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10)
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/precision_recall_tradeoff.png', dpi=300, bbox_inches='tight')
    print(f"Αποθηκεύτηκε το αρχείο: {output_dir}/precision_recall_tradeoff.png")
    plt.close()
    
    #γραφημα χρονου εκτελεσης
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(models))
    bars = ax.bar(x_pos, times, color=['steelblue', 'coral', 'seagreen', 'purple'], alpha=0.7, edgecolor='black')
    ax.set_xlabel('Μοντέλα', fontsize=12)
    ax.set_ylabel('Χρόνος (δευτερόλεπτα)', fontsize=12)
    ax.set_title('Σύγκριση Χρόνου Εκτέλεσης', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)   #Οριζω τις θεσεις πρωτα
    ax.set_xticklabels(models, rotation=15, ha='right')  #και μετα τα lebels
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}s', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/execution_time.png', dpi=300, bbox_inches='tight')
    print(f"Αποθηκεύτηκε το αρχείο: {output_dir}/execution_time.png")
    plt.close()


def save_results_to_file(comparison_data, best_param, output_file='results/results4.txt'):
    #Αποθήκευση αποτελεσμάτων σε αρχείο κειμένου
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("ΑΠΟΤΕΛΕΣΜΑΤΑ ΕΡΩΤΗΜΑΤΟΣ 4 - ΣΥΓΚΡΙΣΗ ΜΟΝΤΕΛΩΝ\n")
        f.write("="*70 + "\n\n")
        
        f.write("ΒΕΛΤΙΣΤΕΣ ΠΑΡΑΜΕΤΡΟΙ SKLEARN TF-IDF:\n")
        f.write("-"*70 + "\n")
        for key, value in best_param.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write("ΑΠΟΤΕΛΕΣΜΑΤΑ ΣΥΓΚΡΙΣΗΣ:\n")
        f.write("-"*85 + "\n")
        f.write(f"{'Μοντέλο':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Precision@10':<12} {'Time':<12}\n")
        f.write("-"*85 + "\n")
        
        for data in comparison_data:
            f.write(f"{data['model']:<20} {data['precision']:<12.4f} {data['recall']:<12.4f} "
                   f"{data['f1_score']:<12.4f} {data['precision@10']:<12.4f} {data['execution_time']:<12.4f}\n")
        
        f.write("-"*85 + "\n\n")
        
        #παρατηρήσεις
        best_f1 = max(comparison_data, key=lambda x: x['f1_score'])
        fastest = min(comparison_data, key=lambda x: x['execution_time'])
        best_precision = max(comparison_data, key=lambda x: x['precision'])
        best_recall = max(comparison_data, key=lambda x: x['recall'])
        
        f.write("ΠΑΡΑΤΗΡΗΣΕΙΣ:\n")
        f.write("-"*70 + "\n")
        f.write(f"1. Καλύτερο F1-Score: {best_f1['model']} ({best_f1['f1_score']:.4f})\n")
        f.write(f"2. Υψηλότερη Precision: {best_precision['model']} ({best_precision['precision']:.4f})\n")
        f.write(f"3. Υψηλότερο Recall: {best_recall['model']} ({best_recall['recall']:.4f})\n")
        f.write(f"4. Ταχύτερο μοντέλο: {fastest['model']} ({fastest['execution_time']:.4f}s)\n\n")
        
        f.write("ΣΥΜΠΕΡΑΣΜΑΤΑ:\n")
        f.write("-"*70 + "\n")
        f.write("- Το Probabilistic μοντέλο προσφέρει την καλύτερη ισορροπία (F1-Score)\n")
        f.write("- Το Sklearn TF-IDF είναι εξαιρετικά γρήγορο αλλά με χαμηλό Recall\n")
        f.write("- Τα TF-IDF/Log TF-IDF έχουν πολύ υψηλό Recall αλλά χαμηλή Precision\n")
        f.write("- Υπάρχει σαφές trade-off μεταξύ Precision και Recall\n")
    
    print(f"\nΑποθηκεύτηκε: {output_file}")
    

if __name__ == "__main__":
    print("Φόρτωση δεδομένων........")
    docs = load_documents("data/docs")
    queries = load_queries("data/queries.txt")
    relevant = load_relevant("data/relevant.txt")
    relevant = filtering_missing_docs(relevant, docs)

    print(f"Έγγραφα: {len(docs)}")
    print(f"Ερωτήματα: {len(queries)}")
    print(f"Σχετικά έγγραφα: {sum(len(v) for v in relevant.values())}")

    #Δημιουργία αναστραμμένου ευερετηρίου για τα μοντέλα του ερωτήματοος 2
    print("\nΔημιουργία inverted index...............")
    inverted_index = create_inverted_index(docs)
    idf = calculate_idf(inverted_index, len(docs))
    prob_idf = calculate_paper_prob_idf(inverted_index, len(docs))

    #Παραμετροποίηση TfidfVectorizer
    print("\n" + "="*70)
    print("ΕΝΑΡΞΗ ΠΑΡΑΜΕΤΡΟΠΟΙΗΣΗΣ TfidfVectorizer")
    print("="*70)
    results_list, best_param = test_tfidf_parameters(queries, docs, relevant)

    #Σύγκριση όλων των μοντέλων
    comparison_data = compare_all_models_used(queries, docs, relevant, inverted_index, idf, prob_idf, best_param)

    # Δημιουργία γραφημάτων
    print("\nΔημιουργία γραφημάτων...")
    visualization_of_data(comparison_data)
    
    # Αποθήκευση αποτελεσμάτων
    save_results_to_file(comparison_data, best_param)
    
    print("\n" + "="*70)
    print("\nΟλοκληρώθηκε η ανάλυση!")
    print("\nΑρχεία που δημιουργήθηκαν:")
    print("  - results/results4.txt")
    print("  - results/precision_recall_tradeoff.png")
    print("  - results/execution_time.png")




