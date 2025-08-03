import os
import re
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

# =======================
# Utility Functions
# =======================

def ensure_dir(directory):
    """Create directory if not exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def clean_filename(name: str) -> str:
    """Clean system/disease names to safe filenames."""
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)

def custom_tokenizer(text):
    """Tokenizer to keep HPO IDs intact."""
    return re.findall(r'HP:\d{7}', text)

# =======================
# Data Loading
# =======================

def load_hpo_graph(obo_path="./hpo_data/hp.obo"):
    """Parses the HPO OBO file into a directed graph."""
    G = nx.DiGraph()
    with open(obo_path) as f:
        current_term = None
        for line in f:
            line = line.strip()
            if line == "[Term]":
                current_term = None
            elif line.startswith("id: HP:"):
                current_term = line.split()[1]
                G.add_node(current_term)
            elif line.startswith("is_a: HP:"):
                parent = line.split()[1]
                if current_term:
                    G.add_edge(parent, current_term)
    return G

def build_hpo_name_map(obo_path="./hpo_data/hp.obo"):
    """Builds dictionary mapping HPO ID to its name."""
    name_map = {}
    with open(obo_path) as f:
        current_id = None
        for line in f:
            line = line.strip()
            if line.startswith("id: HP:"):
                current_id = line.split()[1]
            elif line.startswith("name:") and current_id:
                name_map[current_id] = line[6:]
    return name_map

def load_hpoa_file(hpoa_path="./hpo_data/phenotype.hpoa"):
    """Load the HPO annotation file (.hpoa) and filter for phenotypic annotations."""
    df = pd.read_csv(hpoa_path, sep="\t", comment="#", low_memory=False)
    df = df[df['aspect'] == 'P']
    return df

# =======================
# Data Transformation
# =======================

def build_disease_to_hpo(df: pd.DataFrame) -> dict:
    disease_to_hpos = defaultdict(set)
    for _, row in df.iterrows():
        disease_to_hpos[row["database_id"]].add(row['hpo_id'])
    return {k: list(v) for k, v in disease_to_hpos.items()}

def build_system_documents(disease_to_hpos, system_roots, G):
    """Build a concatenated string of HPO IDs per system for TF-IDF."""
    hpo_to_system = {}
    for root_id, system_name in system_roots.items():
        for desc in nx.descendants(G, root_id):
            hpo_to_system[desc] = system_name
        hpo_to_system[root_id] = system_name

    system_to_terms = defaultdict(list)
    for disease, hpos in disease_to_hpos.items():
        for hpo in hpos:
            if hpo in hpo_to_system:
                system_to_terms[hpo_to_system[hpo]].append(hpo)

    return {system: " ".join(terms) for system, terms in system_to_terms.items()}

def compute_tfidf(documents: dict) -> pd.DataFrame:
    """
    Computes an optimized TF-IDF matrix from a dict of {item: "HPO terms"}.

    system_documents: dict
        Keys = organ systems or diseases
        Values = single string of HPO terms separated by spaces
    """
    items = list(documents.keys())
    docs = [documents[item] for item in items]
    
    # ⚡ OPTIMIZATION PARAMETERS ⚡
    vectorizer = TfidfVectorizer(
        tokenizer=custom_tokenizer,  # only capture proper HPO IDs
        dtype=np.float32,           # use float32 to cut memory in half
        lowercase=False,             # skip lowercasing since HPO IDs are already uniform
        token_pattern=None
    )
    
    tfidf_matrix = vectorizer.fit_transform(docs)  # keep as sparse for speed/memory

    # Convert to a sparse-friendly DataFrame
    tfidf_df = pd.DataFrame.sparse.from_spmatrix(
        tfidf_matrix, 
        index=items, 
        columns=vectorizer.get_feature_names_out()
    )
    
    return tfidf_df

# =======================
# Visualization Functions
# =======================

def plot_top_terms(tfidf_df, name_map, output_dir="./results", top_n=15, threshold=0.1):
    ensure_dir(output_dir)
    for entity in tfidf_df.index:
        row = tfidf_df.loc[entity]
        top_terms = row.sort_values(ascending=False).head(top_n)
        top_terms = top_terms[top_terms > threshold]
        if top_terms.empty:
            continue

        labels = [name_map.get(hpo, hpo) for hpo in top_terms.index]

        plt.figure(figsize=(8, 6))
        sns.barplot(x=top_terms.values, y=labels, color='skyblue')
        plt.title(f"Top Terms for {entity}")
        plt.xlabel("TF-IDF Score")
        plt.ylabel("HPO Term")
        plt.tight_layout()

        fname = os.path.join(output_dir, f"{clean_filename(entity)}_top_terms.png")
        plt.savefig(fname)
        plt.close()

def plot_clustermap(tfidf_df, output_path, level="system"):
    ensure_dir(os.path.dirname(output_path))

    # Use Ward+Euclidean for small dense data, Average+Cosine for larger data
    if tfidf_df.shape[0] <= 20:
        metric, method = 'euclidean', 'ward'
    else:
        metric, method = 'cosine', 'average'

    sns.clustermap(tfidf_df, metric=metric, method=method, figsize=(12, 8), cmap="viridis")
    plt.savefig(output_path)
    plt.close()

# =======================
# Main Pipeline
# =======================

if __name__ == "__main__":
    output_dir = "./results"
    ensure_dir(output_dir)

    # Load data
    G = load_hpo_graph()
    name_map = build_hpo_name_map()
    hpoa_df = load_hpoa_file()
    disease_to_hpos = build_disease_to_hpo(hpoa_df)

    # Define organ systems
    top_systems = {
        "HP:0000119": "Abnormality of the genitourinary system",
        "HP:0000707": "Abnormality of the nervous system",
        "HP:0000818": "Abnormality of the endocrine system",
        "HP:0001197": "Abnormality of prenatal development or birth",
        "HP:0001574": "Abnormality of the integument",
        "HP:0001626": "Abnormality of the cardiovascular system",
        "HP:0001871": "Abnormality of blood and blood-forming tissues",
        "HP:0001939": "Abnormality of metabolism/homeostasis",
        "HP:0002086": "Abnormality of the respiratory system",
        "HP:0002715": "Abnormality of the immune system",
        "HP:0025031": "Abnormality of the digestive system",
        "HP:0033127": "Abnormality of the musculoskeletal system",
    }

    # Compute TF-IDF for systems
    system_documents = build_system_documents(disease_to_hpos, top_systems, G)
    tfidf_system_df = compute_tfidf(system_documents)

    # Save and visualize
    tfidf_system_df.to_csv(os.path.join(output_dir, "tfidf_systems.tsv"), sep="\t")
    plot_top_terms(tfidf_system_df, name_map, output_dir, top_n=15, threshold=0.1)
    plot_clustermap(tfidf_system_df, os.path.join(output_dir, "system_clustermap.png"), level="system")

    # Compute TF-IDF for diseases
    disease_documents = {d: " ".join(hpos) for d, hpos in disease_to_hpos.items()}
    tfidf_disease_df = compute_tfidf(disease_documents)

    tfidf_disease_df.to_csv(os.path.join(output_dir, "tfidf_diseases.tsv"), sep="\t")
    plot_clustermap(tfidf_disease_df, os.path.join(output_dir, "disease_clustermap.png"), level="disease")

    print("Processing complete. Outputs in ./results/")