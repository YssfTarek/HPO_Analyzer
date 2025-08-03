import networkx as nx
from collections import defaultdict
from collections import Counter
import csv
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

def load_hpo_graph(obo_path="./hpo_data/hp.obo"):
    """
    Parses the HPO OBO file into a directed graph.
    Edges go from parent -> child (following is_a relationships).
    """
    
    G = nx.DiGraph()
    
    with open(obo_path) as f:
        current_term = None
        for line in f:
            line = line.strip()
            if line == "[Term]":
                current_term = None  # Start of a new term
            elif line.startswith("id: HP:"):
                current_term = line.split()[1]  # Get current HPO term ID
                G.add_node(current_term)  # Add term as a node
            elif line.startswith("is_a: HP:"):
                parent = line.split()[1]  # Parent term ID
                if current_term:
                    G.add_edge(parent, current_term)  # Edge from parent -> child
    return G

def build_hpo_name_map(obo_path="./hpo_data/hp.obo"):
    name_map = {}
    with open(obo_path) as f:
        current_id = None
        for line in f:
            line = line.strip()
            if line.startswith("id: HP:"):
                current_id = line.split()[1]
            elif line.startswith("name:"):
                if current_id:
                    name_map[current_id] = line[6:]
    return name_map

def load_hpoa_file(hpoa_path = "./hpo_data/phenotype.hpoa"):
    #Load the HPO annotation file (.hpoa) and return a filtered DataFrame with only phenotype annotations.
    df = pd.read_csv(hpoa_path, sep="\t", comment="#", low_memory=False)
    print(df.columns.tolist())
    df = df[df['aspect'] == 'P']  # Keep only phenotypic abnormality annotations
    return df

def build_disease_to_hpo(df:pd.DataFrame) -> dict:
    #Build a dictionary mapping disease IDs to lists of HPO terms.
    disease_to_hpos = defaultdict(set)
    
    for _, row, in df.iterrows():
        disease_id = row["database_id"]
        hpo_id = row['hpo_id']
        disease_to_hpos[disease_id].add(hpo_id)
        
    #convert set to list
    return {k: list(v) for k, v in disease_to_hpos.items()}

def build_disease_hpo_matrix(disease_to_hpos: dict) -> pd.DataFrame:
    #Build a binary matrix (DataFrame) where rows are diseases and columns are HPO terms.
    all_terms = set(term for terms in disease_to_hpos.values() for term in terms)
    all_terms = sorted(all_terms) #ensure consistent ordering
    
    data = []
    index = []
    
    for disease_id, hpo_terms in disease_to_hpos.items():
        row = [1 if term in hpo_terms else 0 for term in all_terms]
        data.append(row)
        index.append(disease_id)
        
    df_matrix = pd.DataFrame(data, index=index, columns=all_terms)
    return df_matrix

def apply_tfidf(df_matrix: pd.DataFrame) -> pd.DataFrame:
    transformer = TfidfTransformer()
    tfidf_matrix = transformer.fit_transform(df_matrix.values)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df_matrix.index, columns=df_matrix.columns)
    return tfidf_df

def get_top_terms_for_disease(tfidf_df: pd.DataFrame, disease_id: str, top_n: int = 10) -> list:
    if disease_id not in tfidf_df.index:
        raise ValueError(f"Disease ID {disease_id} not found.")
    
    scores = tfidf_df.loc[disease_id]
    top_terms = scores.sort_values(ascending=False).head(top_n)
    return list(top_terms.index), list(top_terms.values)

def build_hpo_to_system(G, system_roots):
    #map diseases to organ system using the hpo terms enriched for that diseases
    hpo_to_system = {}

    for root_id, system_name in system_roots.items():
        descendants = nx.descendants(G, root_id)
        for hpo_id in descendants:
            hpo_to_system[hpo_id] = system_name
        # Also include the root itself
        hpo_to_system[root_id] = system_name

    return hpo_to_system

def assign_diseases_to_systems(disease_to_hpos, hpo_to_system):
    disease_to_system = {}

    for disease_id, hpo_terms in disease_to_hpos.items():
        system_counter = Counter()
        for hpo in hpo_terms:
            system = hpo_to_system.get(hpo)
            if system:
                system_counter[system] += 1

        if system_counter:
            # Assign the system with the most matching HPO terms
            disease_to_system[disease_id] = system_counter.most_common(1)[0][0]

    return disease_to_system

def group_diseases_by_system(disease_to_system: dict) -> dict:
    system_to_diseases = defaultdict(list)
    for disease_id, system in disease_to_system.items():
        system_to_diseases[system].append(disease_id)
    return system_to_diseases

##per system prioritisation
def build_system_documents(system_to_diseases: dict, disease_to_hpos: dict) -> dict:
    system_documents = {}

    for system, diseases in system_to_diseases.items():
        all_terms = []
        for disease in diseases:
            all_terms.extend(disease_to_hpos.get(disease, []))
        system_documents[system] = " ".join(all_terms)

    return system_documents

def custom_tokenizer(text):
    # This regex matches 'HP:' followed by exactly 7 digits as one token
    return re.findall(r'HP:\d{7}', text)

def compute_tfidf(system_documents: dict) -> pd.DataFrame:
    systems = list(system_documents.keys())
    docs = [system_documents[sys] for sys in systems]
    
    vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, lowercase=False, token_pattern=None)
    tfidf_matrix = vectorizer.fit_transform(docs)

    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=systems, columns=vectorizer.get_feature_names_out())
    return tfidf_df

def get_top_terms_per_system(tfidf_df: pd.DataFrame, top_n: int = 15) -> dict:
    top_terms = {}

    for system in tfidf_df.index:
        row = tfidf_df.loc[system]
        top = row.sort_values(ascending=False).head(top_n)
        top_terms[system] = list(top.index)

    return top_terms

def map_hpo_ids_to_labels(hpo_graph, hpo_ids):
    """
    Given an HPO graph (networkx DiGraph) and a list of HPO IDs,
    return a dictionary mapping each ID to its label (name attribute).
    """
    id_to_label = {}
    for hpo_id in hpo_ids:
        if hpo_id in hpo_graph.nodes:
            label = hpo_graph.nodes[hpo_id].get('name', 'Unknown label')
            id_to_label[hpo_id] = label
        else:
            id_to_label[hpo_id] = 'ID not found in graph'
    return id_to_label

def get_top_terms_per_entity(tfidf_df: pd.DataFrame, top_n=10):
    top_terms = {}
    for entity in tfidf_df.index:
        # Sort terms by TF-IDF score descending for this entity
        sorted_terms = tfidf_df.loc[entity].sort_values(ascending=False)
        # Take top N terms
        top_terms[entity] = sorted_terms.head(top_n).index.tolist()
    return top_terms

#per disease prioritization
def build_disease_documents(disease_to_hpos):
    # disease_to_hpos: dict where key=disease_id, value=list of HPO terms (e.g. "HP:0001622")
    disease_documents = {}
    for disease, hpos in disease_to_hpos.items():
        # Join HPO terms with space so vectorizer treats each as a token
        disease_documents[disease] = " ".join(hpos)
    return disease_documents

def compute_tfidf(documents):
    vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, lowercase=False, token_pattern=None)
    systems = list(documents.keys())
    docs = [documents[sys] for sys in systems]
    tfidf_matrix = vectorizer.fit_transform(docs)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=systems, columns=vectorizer.get_feature_names_out())
    return tfidf_df

def get_top_terms_per_entity(tfidf_df, top_n=10):
    top_terms_per_entity = {}
    for entity in tfidf_df.index:
        row = tfidf_df.loc[entity]
        top_terms = row.sort_values(ascending=False).head(top_n)
        top_terms_per_entity[entity] = list(zip(top_terms.index, top_terms.values))
    return top_terms_per_entity

G = load_hpo_graph()
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())
print("Is DAG (acyclic)?", nx.is_directed_acyclic_graph(G))

roots = [n for n in G.nodes if G.in_degree(n) == 0]
print("Root terms:", roots[:5])

top_level_terms = list(G.successors("HP:0000118"))
print("Top-level terms under 'Phenotypic abnormality':", top_level_terms)

name_map = build_hpo_name_map()

for term in top_level_terms:
    print(f"{term} → {name_map.get(term)}")
    
#Explore Children of a Category (e.g., Skeletal System)
skeletal = "HP:0000118"  # phenotypic abnormality
children = list(G.successors(skeletal)) #this gets one level only can also be (predecessors)
for child in children:
    print(f"{child} → {name_map.get(child)}")


descendants = nx.descendants(G, skeletal) #this traverses down the entire tree can also be (ancestors)
print(f"Skeletal subtree size: {len(descendants)} terms")

#Find all ancestors of a specific phenotype (e.g., bowing of long bones)
phenotype = "HP:0003025"  # Bowing of long bones
ancestors = nx.ancestors(G, phenotype)
for a in sorted(ancestors):
    print(f"{a} → {name_map.get(a)}")
    
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

for hpo_id, label in top_systems.items():
    print(f"{hpo_id} → {name_map[hpo_id]}")
    
# Create a dictionary to map HPO ID → system name
group_map = {}

for system_id, system_label in top_systems.items():
    descendants = nx.descendants(G, system_id)
    for term in descendants:
        group_map[term] = system_label

group_counts = Counter(group_map.values())
for group, count in group_counts.items():
    print(f"{group}: {count} terms")

# Build a list of rows
disease_to_hpos = []

for hpo_id, group in group_map.items():
    term_name = name_map.get(hpo_id, "Unknown")
    disease_to_hpos.append((hpo_id, term_name, group))

with open("./results/hpo_grouped_terms.tsv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["HPO_ID", "Name", "System_Group"])
    writer.writerows(disease_to_hpos)

hpoa_df = load_hpoa_file()

disease_to_hpos = build_disease_to_hpo(hpoa_df)

sample_keys = list(disease_to_hpos.keys())[:3]

for disease_id in sample_keys:
    print(f"Disease: {disease_id}")
    print("HPO Terms:", disease_to_hpos[disease_id])

hpo_matrix = build_disease_hpo_matrix(disease_to_hpos)
print(hpo_matrix.shape)
print(hpo_matrix.head())

tfidf = apply_tfidf(hpo_matrix)

#top_terms, top_scores = get_top_terms_for_disease(tfidf, "OMIM:613060", top_n=10)
group_tfidf = tfidf.loc["OMIM:613060"]
top_terms = group_tfidf[group_tfidf > 0].sort_values(ascending=False).head(10).index.tolist()
top_scores = group_tfidf[group_tfidf > 0].sort_values(ascending=False).head(10).values


for term, score in zip(top_terms, top_scores):
    print(f"{term}: {score:.4f}")
    
hpo_to_system = build_hpo_to_system(G, top_systems)
disease_to_system = assign_diseases_to_systems(disease_to_hpos, hpo_to_system)

system_to_diseases = group_diseases_by_system(disease_to_system)
system_documents = build_system_documents(system_to_diseases, disease_to_hpos)

for system, doc in system_documents.items():
    print(f"{system}: {doc[:100]}")  # print first 100 chars

tfidf_df = compute_tfidf(system_documents)
top_terms_per_system = get_top_terms_per_system(tfidf_df, top_n=15)
print(top_terms_per_system)


##one group
for hpo_id, score in zip(top_terms, top_scores):
    print(f"{hpo_id} ({name_map.get(hpo_id, 'Unknown')}): {score:.4f}")
    
#per system:
'''for system, hpo_ids in top_terms_per_system.items():
    print(f"System: {system}")
    for hpo_id in hpo_ids:
        print(f"  {hpo_id} ({name_map.get(hpo_id, 'Unknown')})")
    print()'''
    
#per system scored:
threshold = 0.125

for system, hpo_ids in top_terms_per_system.items():
    print(f"System: {system}")
    for hpo_id in hpo_ids:
        if hpo_id in tfidf_df.columns:
            score = tfidf_df.loc[system, hpo_id]
            if score > threshold:
                print(f"  {hpo_id} ({name_map.get(hpo_id, 'Unknown')}): {score:.4f}")
    print()
    
#per disease exploration:
disease_documents = build_disease_documents(disease_to_hpos)
tfidf_disease_df = compute_tfidf(disease_documents)
top_terms_per_disease = get_top_terms_per_entity(tfidf_disease_df, top_n=10)

# Print example for one disease
for hpo_id, score in top_terms_per_disease['OMIM:100800']:
    print(f"{hpo_id} ({name_map.get(hpo_id, 'Unknown')}): {score:.4f}")