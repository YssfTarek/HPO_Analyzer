import os
import re
from collections import Counter
from docx import Document
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
import matplotlib.pyplot as plt
from adjustText import adjust_text

# Path where docx data is stored
FOLDER_PATH = "/Users/youssef/Documents/HPO_Analyzer/data"

# Output folder for saving results
OUTPUT_FOLDER = "./results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Step 1: Extract text from a Word file
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return '\n'.join([para.text for para in doc.paragraphs])

# Step 2: Extract HPO terms and codes as (term, code) tuples from text
def extract_terms(text):
    # regex
    pattern = re.compile(r'([A-Za-z,\-\s]+)(HP:\d+)', re.UNICODE)
    matches = pattern.findall(text)
    matches = [(term.strip(), code) for term, code in matches]
    return set(matches)

# Step 3a: Collect all patient term sets from files
def get_patient_term_sets(folder_path):
    patient_term_sets = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.docx'):
            full_path = os.path.join(folder_path, filename)
            text = extract_text_from_docx(full_path)
            terms = extract_terms(text)
            if terms:
                patient_term_sets.append(terms)
    return patient_term_sets

# Step 3b: Calculate plain frequency of each individual term across all patients
def get_term_frequencies(patient_term_sets):
    all_terms = []
    for patient_terms in patient_term_sets:
        all_terms.extend(patient_terms)
    term_counts = Counter(all_terms)
    return term_counts

# Step 3c: Build transaction DataFrame for Apriori algorithm
def build_transaction_df(patient_term_sets):
    all_terms = sorted({term for patient_terms in patient_term_sets for term, code in patient_terms})
    data = []
    for patient_terms in patient_term_sets:
        terms_in_patient = {term for term, code in patient_terms}
        row = [term in terms_in_patient for term in all_terms]
        data.append(row)
    df = pd.DataFrame(data, columns=all_terms)
    return df

# Step 3d: Apply Apriori algorithm to find frequent itemsets of any size
def find_frequent_itemsets(df, min_support=0.1):
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)
    return frequent_itemsets

# Step 4: Generate association rules
def generate_rules(frequent_itemsets, metric="confidence", min_threshold=0.6):
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
    return rules.sort_values(by=metric, ascending=False)

# Step 5: Visualize association rules as a network and save image
def visualize_rules(rules, max_rules=20, save_path=None):
    G = nx.DiGraph()
    selected_rules = rules.head(max_rules)

    for _, row in selected_rules.iterrows():
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        confidence = row['confidence']
        lift = row['lift']

        for ant in antecedents:
            for cons in consequents:
                G.add_node(ant, role='antecedent')
                G.add_node(cons, role='consequent')
                G.add_edge(ant, cons, weight=confidence, label=f"conf: {confidence:.2f}\nlift: {lift:.2f}")

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=2.0, seed=42)

    node_colors = []
    for node in G.nodes():
        if G.nodes[node].get('role') == 'antecedent':
            node_colors.append('#1f78b4')  # Blue
        else:
            node_colors.append('#ff7f00')  # Orange

    degrees = dict(G.degree())
    node_sizes = [300 + degrees[n]*300 for n in G.nodes()]

    nodes = nx.draw_networkx_nodes(G, pos,
                           node_color=node_colors,
                           node_size=node_sizes,
                           alpha=0.9)

    edges = G.edges(data=True)
    edge_weights = [edge_attr['weight'] for _, _, edge_attr in edges]
    min_w, max_w = min(edge_weights), max(edge_weights)
    edge_widths = [1 + 4*(w - min_w)/(max_w - min_w) if max_w > min_w else 2 for w in edge_weights]

    nx.draw_networkx_edges(G, pos,
                           arrowstyle='-|>',
                           arrowsize=20,
                           edge_color='gray',
                           width=edge_widths,
                           alpha=0.7)

    texts = []
    for node, (x, y) in pos.items():
        texts.append(plt.text(x, y, node, fontsize=10, fontweight='bold'))

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Add legend
    import matplotlib.patches as mpatches
    antecedent_patch = mpatches.Patch(color='#1f78b4', label='Antecedent')
    consequent_patch = mpatches.Patch(color='#ff7f00', label='Consequent')
    plt.legend(handles=[antecedent_patch, consequent_patch], loc='best', fontsize=12)

    plt.title("HPO Term Association Rules Network", fontsize=14)
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Network graph saved to {save_path}")

    plt.show()

if __name__ == "__main__":
    # Get terms per patient
    patient_term_sets = get_patient_term_sets(FOLDER_PATH)
    print(f"Extracted terms for {len(patient_term_sets)} patients.\n")
    
    # Plain frequency counts (single terms)
    term_freq = get_term_frequencies(patient_term_sets)
    print("Top 20 most common individual terms:")
    for (term, code), count in term_freq.most_common(20):
        print(f"{term} ({code}): {count}")
    
    # Save term frequencies to CSV
    term_freq_df = pd.DataFrame(term_freq.items(), columns=['(term, code)', 'count'])
    term_freq_df[['term', 'code']] = pd.DataFrame(term_freq_df['(term, code)'].tolist(), index=term_freq_df.index)
    term_freq_df.drop(columns='(term, code)', inplace=True)
    term_freq_df = term_freq_df[['term', 'code', 'count']]
    term_freq_df.sort_values(by='count', ascending=False, inplace=True)
    term_freq_path = os.path.join(OUTPUT_FOLDER, "term_frequencies.csv")
    term_freq_df.to_csv(term_freq_path, index=False)
    print(f"\nTerm frequencies saved to {term_freq_path}")
    
    # Build transaction DataFrame for Apriori
    df_transactions = build_transaction_df(patient_term_sets)
    print(f"\nTransaction DataFrame shape: {df_transactions.shape} (patients x unique terms)")
    
    # Find frequent itemsets with minimum support
    min_support = 0.01
    frequent_itemsets = find_frequent_itemsets(df_transactions, min_support=min_support)
    print(f"\nFrequent itemsets (term sets co-occurring in >= {int(min_support*100)}% of patients):")
    for _, row in frequent_itemsets.head(20).iterrows():
        terms = ', '.join(row['itemsets'])
        print(f"Terms: [{terms}] - Support: {row['support']:.2f}")

    # Generate and display association rules
    rules = generate_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
    print(f"\nTop {min(10, len(rules))} association rules by confidence:")
    for i, row in rules.head(10).iterrows():
        ant = ', '.join(row['antecedents'])
        cons = ', '.join(row['consequents'])
        print(f"{ant} => {cons} (conf: {row['confidence']:.2f}, lift: {row['lift']:.2f})")

    # Now convert itemsets and rules columns to strings for saving
    frequent_itemsets_to_save = frequent_itemsets.copy()
    frequent_itemsets_to_save['itemsets'] = frequent_itemsets_to_save['itemsets'].apply(lambda x: ', '.join(x))
    frequent_itemsets_path = os.path.join(OUTPUT_FOLDER, "frequent_itemsets.csv")
    frequent_itemsets_to_save.to_csv(frequent_itemsets_path, index=False)
    print(f"\nFrequent itemsets saved to {frequent_itemsets_path}")

    rules_to_save = rules.copy()
    rules_to_save['antecedents'] = rules_to_save['antecedents'].apply(lambda x: ', '.join(x))
    rules_to_save['consequents'] = rules_to_save['consequents'].apply(lambda x: ', '.join(x))
    rules_path = os.path.join(OUTPUT_FOLDER, "association_rules.csv")
    rules_to_save.to_csv(rules_path, index=False)
    print(f"\nAssociation rules saved to {rules_path}")

    # Visualize rules and save the plot
    visualize_rules(rules, max_rules=20, save_path=os.path.join(OUTPUT_FOLDER, "association_rules_network.png"))