import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

FOLDER_PATH = "/Users/youssef/Documents/HPO_Analyzer/realistic_patient_terms.csv"

raw_df = pd.read_csv(FOLDER_PATH)

#convert terms to lists instead of strings
raw_df['Terms'] = raw_df['Terms'].apply(lambda x: [term.strip() for term in str(x).split(";")])

#cluster the data by disease group
groups = raw_df.groupby('Group')

#prepare transaction df
transactions = raw_df['Terms']

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_trans = pd.DataFrame(te_array, columns=te.columns_)

#set apriori rules
frequent = apriori(df_trans, min_support=0.05, use_colnames=True)
rules = association_rules(frequent, metric="confidence", min_threshold=0.6)

#print results
print("Frequent itemsets:")
print(frequent.sort_values('support', ascending=False).head())

print("\nTop rules:")
print(rules.sort_values('confidence', ascending=False).head())