import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("data.csv")

# Select relevant numeric + encoded categorical columns
df = df.dropna(subset=["Sentiment"])  # remove null sentiments
df["Sentiment"] = df["Sentiment"].astype('category').cat.codes  # Encode positive/negative/neutral

# Select features (you can add more columns)
features = [
    "st_turnover_crores", "st_delivered_value_crores", "st_total",
    "traded_contract_lots", "total_value_lacs", "avg_daily_turnover_lacs",
    "Confidence"
]
X = df[features]
y = df["Sentiment"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
