import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# --- Step 1: Load dataset ---
df = pd.read_csv("data.csv")

# --- Step 2: Check available columns ---
print(" Available Columns in Data:")
print(df.columns.tolist())

# --- Step 3: Use correct column names (case-insensitive) ---
# Rename to lowercase for consistency
df.columns = df.columns.str.lower()

# Pick available columns (ignore missing ones)
possible_cols = ["instrument_type", "segment", "commodity", "month", "year"]
selected_cols = [col for col in possible_cols if col in df.columns]

df = df[selected_cols].dropna()
print(f"\n Using columns: {selected_cols}")

# --- Step 4: One-hot encode categorical data ---
df_encoded = pd.get_dummies(df.astype(str))
print("\n Data Prepared for Association Rule Mining")
print(df_encoded.head())

# --- Step 5: Apply Apriori ---
frequent_items = apriori(df_encoded, min_support=0.05, use_colnames=True)
print("\n Frequent Itemsets (Top 10):")
print(frequent_items.head(10))

# --- Step 6: Generate Rules ---
rules = association_rules(frequent_items, metric="lift", min_threshold=1.0)
rules = rules.sort_values(by="confidence", ascending=False)

print("\n Top Association Rules:")
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10))

