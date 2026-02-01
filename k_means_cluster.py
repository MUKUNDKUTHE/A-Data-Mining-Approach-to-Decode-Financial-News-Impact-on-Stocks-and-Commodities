import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt

# --- Step 1: Load data ---
df = pd.read_csv("data.csv")

# --- Step 2: Select new unused columns ---
features = [
    "st_no_of_trades_lacs", "st_traded_qty_lacs",
    "st_delivered_qty_lacs", "st_perc_dlvrd_to_traded_qty",
    "st_short_dlvry_auc_qty_lacs", "st_short_dlvry_value",
    "st_funds_payin_crores"
]

df_selected = df[features].dropna()  # remove nulls

# --- Step 3: Standardize the data ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_selected)

# --- Step 4: Apply K-Means clustering ---
kmeans = KMeans(n_clusters=3, random_state=42)
df_selected["Cluster"] = kmeans.fit_predict(X_scaled)

# --- Step 5: Evaluate clustering ---
silhouette = silhouette_score(X_scaled, df_selected["Cluster"])
print(f"Silhouette Score: {silhouette:.3f}")

# --- Step 6: Cluster summary ---
print("\nCluster Summary (Average per Cluster):")
print(df_selected.groupby("Cluster")[features].mean())

# --- Step 7: Visualize the clusters ---
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df_selected,
    x="st_no_of_trades_lacs",
    y="st_delivered_qty_lacs",
    hue="Cluster",
    palette="Set2"
)
plt.title("Clustering of Market Activity Patterns")
plt.tight_layout()
plt.show()
