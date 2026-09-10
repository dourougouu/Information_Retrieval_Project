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

def average_precision(ranked_docs, relevant_docs):
    #μεση ακριβεια για 1 ερωτημα
    relevant_set = set(relevant_docs)
    scr = 0.0
    num_relevant= 0
    for i, doc in enumerate(ranked_docs, 1):
        if doc in relevant_set:
            num_relevant += 1
            scr += num_relevant / i
    return scr/len(relevant_set) if relevant_set else 0.0

def mean_avg_precision(ranked_results, relevant):
    #υπολογισμος MAP για ολα τα ερωτηματα
    aps = []
    for qID, ranked in ranked_results.items():
        if qID in relevant:
            relevant_docs = list(relevant[qID].keys())
            ranked_docs = [doc for doc, rnk in ranked]
            ap = average_precision(ranked_docs, relevant_docs)
            aps.append(ap)
    return np.mean(aps) if aps else 0.0

def preprocess_for_sklearn(text):
    #Απλή προεπεξεργασία κειμένου για χρήση με sklearn. 
    #Το TfidfVectorizer κάνει μόνο του την δική του προεπεξεργασία 
    return text.lower()
    
def sklearn_tfidf_retrieval(queries, docs, vector, k_values=[5, 10, 20]):
    #Ανάκτηση με χρήση του TfidfVectorizer από την sklearn
    #Επιστρέφει τα αποτελέσματα της ανάκτησης (results)
    #Τον χρόνο δεικτοδότησης (indexing_time) και τον συνολικό χρόνο ερωτημάτων (retrieval_time)

    #Προεπεξεργασία εγγράφων
    doc_IDs = list(docs.keys())
    doc_texts = [preprocess_for_sklearn(docs[doc_ID]) for doc_ID in doc_IDs]

    #Δεικτοδότηση με TfidfVectorizer
    start_time = time.time()
    doc_vectors = vector.fit_transform(doc_texts)
    indexing_time = time.time() - start_time

    #Ανάκτηση για κάθε query
    results = {}
    start_retrieval_time = time.time()   
        
    for qID, query_text in queries.items():
        query_vector = vector.transform([preprocess_for_sklearn(query_text)])
        similarities = cosine_similarity(query_vector, doc_vectors)[0]
        ranked = [(doc_IDs[i], similarities[i]) for i in range(len(doc_IDs))] #δημιουργία λίστας (docID, score) και ταξινόμηση
        ranked.sort(key=lambda x: x[1], reverse=True)
        results[qID] = ranked

    retrieval_time = time.time() - start_retrieval_time
    return results, indexing_time, retrieval_time

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
            ranked_results, indexing_time, retrieval_time = sklearn_tfidf_retrieval(queries, docs, vector, k_values)
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
                'f1_score': np.mean([m['f1_score'] for m in all_metrics]),
                'MAP': mean_avg_precision(ranked_results, relevant)
            }
            for k in k_values:
                avg_metrics[f'precision@{k}'] = np.mean([m[f'precision@{k}'] for m in all_metrics])

            #αποθήκευση αποτελεσμάτων 
            results={
                'param': param,
                'indexing_time': indexing_time,
                'retrieval_time': retrieval_time,
                **avg_metrics
            }
            results_list.append(results)

            print(f" F1-Score: {avg_metrics['f1_score']:.4f}, MAP: {avg_metrics['MAP']:.4f}")

        except Exception as e:
            print(f"ΣΦΑΛΜΑ: {e}")
            continue

    #Εύρεση των βέλτιστων παραμέτρων με βάση το F1-Score
    best_idx = max(range(len(results_list)), key=lambda i:results_list[i]['f1_score'])
    best_result = results_list[best_idx]
    best_param = best_result['param']

    print("\n" + "="*70)
    print("ΒΕΛΤΙΣΤΕΣ ΠΑΡΑΜΕΤΡΟΙ:")
    for key, value in best_param.items():
        print(f"   {key}: {value}")
    print(f"\nΑποτελέσματα:")
    print(f"   Precision: {best_result['precision']:.4f}")
    print(f"   Recall: {best_result['recall']:.4f}")
    print(f"   F1-Score: {best_result['f1_score']:.4f}")
    print(f"   MAP: {best_result['MAP']:.4f}")
    print(f"   Precision@10: {best_result['precision@10']:.4f}")
    print("="*70)
    return results_list, best_param

def analyze_best_parameters(best_param, results_list):
    #Αναλύει γιατί οι βέλτιστες παράμετροι βελτιώνουν την απόδοση
    
    analysis = []
    
    if best_param['ngram_range'] == (1, 2):
        analysis.append("- Bigrams (1,2): Αποτυπώνουν φράσεις που εχουν σημασία όπως 'cystic fibrosis'")
    elif best_param['ngram_range'] == (1, 3):
        analysis.append("- Trigrams (1,3): Μεγαλύτερο context αλλά περισσότερος θόρυβος")
    else:
        analysis.append("- Unigrams (1,1): Απλή αναπαράσταση με επιμέρους όρους")

    if best_param['sublinear_tf']:
        analysis.append("- sublinear_tf=True: Μειώνει την επίδραση πολύ συχνών όρων εφαρμόζωντας λογαριθμική στάθμιη (log scaling)")
    
    if best_param['min_df'] > 1:
        analysis.append(f"- min_df={best_param['min_df']}: Αφαιρεί σπάνιους/θορυβώδεις όρους")
    
    if best_param['max_df'] < 1.0:
        analysis.append(f"- max_df={best_param['max_df']}: Αφαιρεί/Αγνοεί πολύ συχνούς όρους (σαν stopwords)")
    
    if best_param['norm'] == 'l2':
        analysis.append("- norm='l2': Κανονικοποίηση Euclidean - τυπική για cosine similarity")
    elif best_param['norm'] == 'l1':
        analysis.append("- norm='l1': Κανονικοποίηση Manhattan")
    else:
        analysis.append("- norm=None: Χωρίς κανονικοποίηση")
    
    return "\n".join(analysis)


def compare_all_models_used(queries, docs, relevant, inverted_index, idf, prob_idf, best_param, k_values=[5, 10,20]):
    print("\n" + "="*70)
    print("ΣΥΓΚΡΙΣΗ ΟΛΩΝ ΤΩΝ ΜΟΝΤΕΛΩΝ")
    print("="*70)
    indexing_time_tfidf = 0 #χρονος δεικτοδοτησης (Θα μετρηθει αργοτερα στη main)
    start_time = time.time()
    models_results={}

    #1 TF-IDF από ερώτημα 2
    print("\n1. Υπολογισμός για TF-IDF (Ερώτημα 2)...")
    start_retrieval = time.time() #για χρονο ανακτησης
    tfidf_results = {}
    for qID, query_text in queries.items():
        ranked = ranking_docs(query_text, docs, inverted_index, idf, "tfidf")
        tfidf_results[qID] = ranked
    tfidf_retrieval_time = time.time() - start_retrieval #χρονος ανακτησης
    models_results['TF-IDF'] = {
        'results': tfidf_results,
        'retrieval_time': tfidf_retrieval_time
    }

    #2 Log TF-IDF από ερώτημα 2
    print("2. Υπολογισμός για Log TF-IDF (Ερώτημα 2)...")
    start_retrieval = time.time()
    log_results = {}
    for qID, query_text in queries.items():
        ranked = ranking_docs(query_text, docs, inverted_index, idf, "log")
        log_results[qID] = ranked
    log_retrieval_time = time.time() - start_retrieval
    models_results['Log TF-IDF'] = {
        'results': log_results,
        'retrieval_time': log_retrieval_time
    }

    #3 Probabilistic από ερώτημα 2
    print("3. Υπολογισμός για Probabilistic (Ερώτημα 2)...")
    start_retrieval = time.time()
    prob_results = {}
    for qID, query_text in queries.items():
        ranked = ranking_docs(query_text, docs, inverted_index, idf, "probabilistic", prob_idf)
        prob_results[qID] = ranked
    prob_retrieval_time = time.time() - start_retrieval
    models_results['Probabilistic'] = {
        'results': prob_results,
        'retrieval_time': prob_retrieval_time
    }

    #4 Sklearn TF-IDF με βέλτιστες παραμέτρους
    print("4. Υπολογισμός για Sklearn TF-IDF (βέλτιστες παράμετροι)")
    vector = TfidfVectorizer(**best_param, stop_words='english')
    sklearn_results, sklearn_indexing_time, sklearn_retrieval_time = sklearn_tfidf_retrieval(queries, docs, vector, k_values)
    models_results['Sklearn TF-IDF'] = {
        'results': sklearn_results,
        'indexing_time': sklearn_indexing_time,
        'retrieval_time': sklearn_retrieval_time
    }

    indexing_time_manually = time.time() - start_time

    #Υπολογισμός μετρικών για κάθε μονλέλο
    print("\n" + "-"*70)
    print(f"{'Μοντέλο':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'MAP':<12} {'P@5':<10} {'P@10':<10} {'P@20':<10} {'Index(s)':<12} {'Retr(s)':<12} {'Total(s)':<12}")
    print("-"*70)

    comparison_data=[]

    for model_name, model_data in models_results.items():
        all_metrics=[]
        map_scores=[]

        for qID in queries.keys():
            if qID in relevant and qID in model_data['results']:
                relevant_docs = list(relevant[qID].keys())
                metrics = evaluate_model(model_data['results'][qID], relevant_docs, k_values)
                all_metrics.append(metrics)

                ranked_docs = [doc for doc, rnk in model_data['results'][qID]]
                ap = average_precision(ranked_docs, relevant_docs)
                map_scores.append(ap)


        avg_precision = np.mean([m['precision'] for m in all_metrics]) if all_metrics else 0.0
        avg_recall = np.mean([m['recall'] for m in all_metrics]) if all_metrics else 0.0
        avg_f1 = np.mean([m['f1_score'] for m in all_metrics]) if all_metrics else 0.0
        avg_map = np.mean(map_scores) if map_scores else 0.0
        avg_precisionat10 = np.mean([m['precision@10'] for m in all_metrics]) if all_metrics else 0.0
        avg_precisionat5 = np.mean([m['precision@5'] for m in all_metrics]) if all_metrics else 0.0
        avg_precisionat20 = np.mean([m['precision@20'] for m in all_metrics]) if all_metrics else 0.0

        if model_name == 'Sklearn TF-IDF':
            indexing_time = model_data['indexing_time']
            retrieval_time = model_data['retrieval_time']
        else:
            indexing_time = indexing_time_manually
            retrieval_time = model_data['retrieval_time']
        
        total_time = indexing_time + retrieval_time
        print(f"{model_name:<20} {avg_precision:<12.4f} {avg_recall:<12.4f} {avg_map:<12.4f} {avg_f1:<12.4f} {avg_precisionat5:<10.4f} {avg_precisionat10:<10.4f} {avg_precisionat20:<10.4f} {indexing_time:<12.4f} {retrieval_time:<12.4f} {total_time:<12.4f}")

        comparison_data.append({
            'model': model_name,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': avg_f1,
            'MAP': avg_map,
            'precision@5': avg_precisionat5,
            'precision@10': avg_precisionat10, 
            'precision@20': avg_precisionat20, 
            'indexing_time': indexing_time,
            'retrieval_time': retrieval_time,
            'execution_time': total_time
        })
    print("-"*70)
    return comparison_data



def visualization_of_data(comparison_data, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    
    models = [d['model'] for d in comparison_data]
    
    # 1. Precision-Recall Trade-off
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['steelblue', 'coral', 'seagreen', 'purple']
    for i, data in enumerate(comparison_data):
        ax.scatter(data['recall'], data['precision'], s=300, alpha=0.6,
                   color=colors[i], label=data['model'], edgecolors='black', linewidths=1.5)
        ax.annotate(data['model'], (data['recall'], data['precision']),
                   textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10)
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Trade-off ανά Μοντέλο', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/precision_recall_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()
    

    # 2. Precision@k Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.25
    p5 = [d['precision@5'] for d in comparison_data]
    p10 = [d['precision@10'] for d in comparison_data]
    p20 = [d['precision@20'] for d in comparison_data]
    
    bars1 = ax.bar(x - width, p5, width, label='P@5', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x, p10, width, label='P@10', color='coral', alpha=0.8)
    bars3 = ax.bar(x + width, p20, width, label='P@20', color='seagreen', alpha=0.8)
    
    ax.set_xlabel('Μοντέλα', fontsize=12)
    ax.set_ylabel('Precision@k', fontsize=12)
    ax.set_title('Σύγκριση Precision@k ανά Μοντέλο', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/precision_at_k.png', dpi=300, bbox_inches='tight')
    plt.close()
    


    # 3. Χρόνοι εκτέλεσης
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(models))
    indexing_times = [d['indexing_time'] for d in comparison_data]
    retrieval_times = [d['retrieval_time'] for d in comparison_data]
    
    bars_index = ax.bar(x_pos - 0.2, indexing_times, 0.4, label='Δεικτοδότηση', color='steelblue', alpha=0.8)
    bars_retr = ax.bar(x_pos + 0.2, retrieval_times, 0.4, label='Ανάκτηση', color='coral', alpha=0.8)
    
    ax.set_xlabel('Μοντέλα', fontsize=12)
    ax.set_ylabel('Χρόνος(sec)', fontsize=12)
    ax.set_title('Χρόνοι Εκτέλεσης ανά Κατηγορία', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/execution_times_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    


    # 4. F1-Score και MAP
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.35
    f1_scores = [d['f1_score'] for d in comparison_data]
    maps = [d['MAP'] for d in comparison_data]
    
    bars1 = ax.bar(x - width/2, f1_scores, width, label='F1-Score', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, maps, width, label='MAP', color='coral', alpha=0.8)
    
    ax.set_xlabel('Μοντέλα', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Σύγκριση F1-Score και MAP', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/f1_map_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Αποθηκεύτηκαν 4 γραφήματα στο φάκελο '{output_dir}/'")



def save_results_to_file(comparison_data, best_param, results_list, output_file='results/results4.txt'):
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

        # Ανάλυση παραμέτρων
        f.write("ΑΝΑΛΥΣΗ ΒΕΛΤΙΣΤΩΝ ΠΑΡΑΜΕΤΡΩΝ:\n")
        f.write("-"*100 + "\n")
        analysis = analyze_best_parameters(best_param, results_list)
        f.write(analysis + "\n\n")
        
        f.write("ΑΠΟΤΕΛΕΣΜΑΤΑ ΣΥΓΚΡΙΣΗΣ:\n")
        f.write("-"*85 + "\n")
        f.write(f"{'Μοντέλο':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'MAP':<12} {'Precision@5':<10} {'Precision@10':<10} {'Precision@20':<10}  {'Index(s)':<10} {'Retr(s)':<10} {'Total(s)':<10}\n")
        f.write("-"*85 + "\n")
        
        for data in comparison_data:
            f.write(f"{data['model']:<20} {data['precision']:<12.4f} {data['recall']:<12.4f} "
                    f"{data['f1_score']:<12.4f} {data['MAP']:<12.4f} {data['precision@5']:<10.4f} {data['precision@10']:<10.4f} {data['precision@20']:<10.4f}  {data['indexing_time']:<10.4f} {data['retrieval_time']:<10.4f} {data['execution_time']:<10.4f}\n")
        f.write("-"*85 + "\n\n")
        
        #ευρεση καλυτερων μοντελων
        best_f1 = max(comparison_data, key=lambda x: x['f1_score'])
        best_map = max(comparison_data, key=lambda x: x['MAP'])
        fastest_total = min(comparison_data, key=lambda x: x['execution_time'])
        fastest_retrieval = min(comparison_data, key=lambda x: x['retrieval_time'])
        best_precision = max(comparison_data, key=lambda x: x['precision'])
        best_recall = max(comparison_data, key=lambda x: x['recall'])
        
        f.write("ΠΑΡΑΤΗΡΗΣΕΙΣ:\n")
        f.write("-"*70 + "\n")
        f.write("ΚΑΤΑΤΑΞΗ ΜΟΝΤΕΛΩΝ:\n")
        f.write("-"*30 + "\n")
        f.write(f"1. Καλύτερο F1-Score: {best_f1['model']} ({best_f1['f1_score']:.4f})\n")
        f.write(f"2. Καλύτερο MAP: {best_map['model']} ({best_map['MAP']:.4f})\n")
        f.write(f"3. Υψηλότερη Precision: {best_precision['model']} ({best_precision['precision']:.4f})\n")
        f.write(f"4. Υψηλότερο Recall: {best_recall['model']} ({best_recall['recall']:.4f})\n")
        f.write(f"5. Ταχύτερο μοντέλο συνολικά: {fastest_total['model']} ({fastest_total['execution_time']:.4f}s)\n\n")
        f.write(f"6. Ταχύτερο μοντέλο ανάκτησης: {fastest_retrieval['model']} ({fastest_retrieval['retrieval_time']:.4f}s)\n\n")

        
        f.write("ΣΥΜΠΕΡΑΣΜΑΤΑ:\n")
        f.write("-"*40 + "\n")
        f.write("- Το Probabilistic μοντέλο προσφέρει την καλύτερη ισορροπία  Precision-Recall (F1-Score)\n")
        f.write("- Το Sklearn TF-IDF είναι εξαιρετικά γρήγορο αλλά με χαμηλό Recall\n")
        f.write("- Τα TF-IDF/Log TF-IDF έχουν πολύ υψηλό Recall αλλά χαμηλή Precision\n")
        f.write("- Η παραμετροποίηση του Sklearn βελτιώνει σημαντικά την απόδοση\n")
        f.write("- Υπάρχει σαφές trade-off μεταξύ Precision και Recall\n")
    
    print(f"\nΑποθηκεύτηκε στο: {output_file}")
    

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
    start_time = time.time()
    inverted_index = create_inverted_index(docs)
    idf = calculate_idf(inverted_index, len(docs))
    prob_idf = calculate_paper_prob_idf(inverted_index, len(docs))
    print(f"Χρόνος δεικτοδότησης: {time.time() - start_time:.4f}s")

    #Παραμετροποίηση TfidfVectorizer
    print("\n" + "="*70) 
    print("ΕΝΑΡΞΗ ΠΑΡΑΜΕΤΡΟΠΟΙΗΣΗΣ TfidfVectorizer")
    print("="*70)
    results_list, best_param = test_tfidf_parameters(queries, docs, relevant)

    #Σύγκριση όλων των μοντέλων
    comparison_data =  compare_all_models_used(queries, docs, relevant, inverted_index, idf, prob_idf, best_param)

    # Δημιουργία γραφημάτων
    print("\nΔημιουργία γραφημάτων...")
    visualization_of_data(comparison_data)
    
    # Αποθήκευση αποτελεσμάτων
    save_results_to_file(comparison_data, best_param, results_list)
    
    print("\n" + "="*70)
    print("\nΟλοκληρώθηκε η ανάλυση!")
    print("\nΑρχεία που δημιουργήθηκαν:")
    print("  • results/results4.txt")
    print("  • results/precision_recall_tradeoff.png")
    print("  • results/precision_at_k.png")
    print("  • results/execution_times_breakdown.png")
    print("  • results/f1_map_comparison.png")
    



