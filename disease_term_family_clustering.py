import os
import re
import logging
from typing import Optional
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

# =======================
# Configuration
# =======================
OBO_PATH = "./hpo_data/hp.obo"
HPOA_PATH = "./hpo_data/phenotype.hpoa"
OUTPUT_DIR = "./results"

TOP_SYSTEMS = {
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

# =======================
# Logging Setup
# =======================
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# =======================
# Utility Functions
# =======================

def ensure_dir(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)

def clean_filename(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)

def custom_tokenizer(text: str) -> List[str]:
    return re.findall(r'HP:\d{7}', text)

# =======================
# Data Loading
# =======================

def load_hpo_graph(obo_path: str = OBO_PATH) -> nx.DiGraph:
    G = nx.DiGraph()
    try:
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
        logger.info(f"Loaded HPO graph with {len(G.nodes)} nodes and {len(G.edges)} edges.")
    except FileNotFoundError:
        logger.error(f"OBO file not found at {obo_path}")
        raise
    return G

def build_hpo_name_map(obo_path: str = OBO_PATH) -> Dict[str, str]:
    name_map = {}
    with open(obo_path) as f:
        current_id = None
        for line in f:
            line = line.strip()
            if line.startswith("id: HP:"):
                current_id = line.split()[1]
            elif line.startswith("name:") and current_id:
                name_map[current_id] = line[6:]
    logger.info(f"Built HPO name map with {len(name_map)} entries.")
    return name_map

def load_hpoa_file(hpoa_path: str = HPOA_PATH) -> pd.DataFrame:
    df = pd.read_csv(hpoa_path, sep="\t", comment="#", low_memory=False)
    df = df[df['aspect'] == 'P']
    logger.info(f"Loaded HPOA file with {len(df)} phenotypic annotations.")
    return df

# =======================
# Data Transformation
# =======================

def build_disease_to_hpo(df: pd.DataFrame) -> Dict[str, List[str]]:
    disease_to_hpos = defaultdict(set)
    for _, row in df.iterrows():
        disease_to_hpos[row["database_id"]].add(row['hpo_id'])
    result = {k: list(v) for k, v in disease_to_hpos.items()}
    logger.info(f"Built disease→HPO map for {len(result)} diseases.")
    return result

def build_system_documents(
    disease_to_hpos: Dict[str, List[str]], 
    system_roots: Dict[str, str], 
    G: nx.DiGraph
) -> Dict[str, str]:
    hpo_to_system = {}
    for root_id, system_name in system_roots.items():
        for desc in nx.descendants(G, root_id):
            hpo_to_system[desc] = system_name
        hpo_to_system[root_id] = system_name

    system_to_terms = defaultdict(list)
    for hpos in disease_to_hpos.values():
        for hpo in hpos:
            if hpo in hpo_to_system:
                system_to_terms[hpo_to_system[hpo]].append(hpo)

    logger.info(f"Built {len(system_to_terms)} system documents.")
    return {system: " ".join(terms) for system, terms in system_to_terms.items()}

def compute_tfidf(documents: Dict[str, str], name: str = "entity", min_df: int = 1, max_df: Optional[float] = 1.0, max_features: int = None, threshold_empty: bool = True) -> pd.DataFrame:
    """
    Compute a TF-IDF matrix for a dictionary of documents.

    Parameters
    ----------
    documents : dict
        Mapping of entity name to space-separated HPO terms.
    name : str
        Label for logging purposes.
    min_df : int
        Minimum number of documents a term must appear in.
    max_features : int
        Maximum number of features to keep.
    threshold_empty : bool
        If True, log a warning and return empty dataframe if no documents.

    Returns
    -------
    pd.DataFrame
        TF-IDF matrix (rows=entities, columns=HPO terms).
    """
    if not documents:
        if threshold_empty:
            logger.warning(f"No {name} documents provided for TF-IDF computation.")
        return pd.DataFrame()

    items = list(documents.keys())
    docs = list(documents.values())

    vectorizer = TfidfVectorizer(
        tokenizer=custom_tokenizer,
        dtype=np.float32,
        lowercase=False,
        token_pattern=None,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features
    )
    
    tfidf_matrix = vectorizer.fit_transform(docs)
    tfidf_df = pd.DataFrame.sparse.from_spmatrix(
        tfidf_matrix,
        index=items,
        columns=vectorizer.get_feature_names_out()
    )
    logger.info(f"{name.capitalize()} TF-IDF matrix computed with shape {tfidf_df.shape}.")
    return tfidf_df

# =======================
# Visualization Functions
# =======================

def plot_top_terms(tfidf_df: pd.DataFrame, name_map: Dict[str, str], output_dir: str = OUTPUT_DIR, top_n: int = 15, threshold: float = 0.1):
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
     
def plot_clustermap(tfidf_df: pd.DataFrame, output_path: str, level: str = "system", tfidf_threshold: int = None, top_n: Optional[int] = None):
    if tfidf_threshold > 0:
        tfidf_df = tfidf_df.loc[:, (tfidf_df > tfidf_threshold).any(axis=0)]
    if tfidf_df.empty:
        logger.warning(f"Skipping {level} clustermap: no data.")
        return
    
    if top_n and top_n > 0 and tfidf_df.shape[0] > top_n:
        tfidf_df = tfidf_df.loc[tfidf_df.sum(axis=1).nlargest(top_n).index]

    ensure_dir(os.path.dirname(output_path))

    metric, method = ('euclidean', 'ward') if tfidf_df.shape[0] <= 20 else ('cosine', 'average')
    sns.clustermap(tfidf_df, metric=metric, method=method, figsize=(12, 8), cmap="viridis")
    plt.savefig(output_path)
    plt.close()

# =======================
# Main Pipeline
# =======================

def main():
    ensure_dir(OUTPUT_DIR)

    G = load_hpo_graph()
    name_map = build_hpo_name_map()
    hpoa_df = load_hpoa_file()
    disease_to_hpos = build_disease_to_hpo(hpoa_df)

    # System-level TF-IDF
    system_documents = build_system_documents(disease_to_hpos, TOP_SYSTEMS, G)
    tfidf_system_df = compute_tfidf(system_documents, name="system",max_df=0.8)
    tfidf_system_df.to_csv(os.path.join(OUTPUT_DIR, "tfidf_systems.tsv"), sep="\t")
    plot_top_terms(tfidf_system_df, name_map, OUTPUT_DIR)
    plot_clustermap(tfidf_system_df, os.path.join(OUTPUT_DIR, "system_clustermap.png"), level="system", tfidf_threshold=0.1)

    # Disease-level TF-IDF (optimized)
    disease_id_to_name = hpoa_df.drop_duplicates('database_id').set_index('database_id')['disease_name'].to_dict()
    disease_documents = {disease_id_to_name.get(d, d): " ".join(hpos) for d, hpos in disease_to_hpos.items()}


    tfidf_disease_df = compute_tfidf(disease_documents, name="disease", min_df=1, max_df=0.8, max_features=500)
    tfidf_disease_df = tfidf_disease_df.loc[~(tfidf_disease_df==0).all(axis=1)]
    tfidf_disease_df = tfidf_disease_df.loc[:, ~(tfidf_disease_df==0).all(axis=0)]
    tfidf_disease_df = tfidf_disease_df.fillna(0)
    tfidf_disease_df.to_csv(os.path.join(OUTPUT_DIR, "tfidf_diseases.tsv"), sep="\t")
    plot_clustermap( tfidf_disease_df, os.path.join(OUTPUT_DIR, "disease_clustermap.png"), level="disease", tfidf_threshold=0.1, top_n=15)

    logger.info("Processing complete. Outputs in 'results' directory.")

if __name__ == "__main__":
    main()