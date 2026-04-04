import pandas as pd
import re
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os


def parse_parameters(param_str):
    params = {}
    parts = param_str.split("|")
    for part in parts:
        part = part.strip()
        if "=" in part:
            key, val = part.split("=", 1)
            val = re.sub(r"Mbps|ms|%|devices/km2|km/h", "", val).strip()
            try:
                params[key.strip()] = float(val)
            except ValueError:
                params[key.strip()] = val.strip()
    return params


print("Loading dataset...")
df_raw = pd.read_csv(
    "/Users/Agentic RAN/network_slicing_300.csv"
)
parsed_data = df_raw["Parameters"].apply(parse_parameters)
df = pd.DataFrame(list(parsed_data))

X = df[["Throughput", "Latency", "Reliability", "Density", "Mobility"]]

print("Training Random Forest Classifier...")
label_enc = LabelEncoder()
y = label_enc.fit_transform(df["Error"])

rf_model = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight="balanced"
)
rf_model.fit(X, y)

print("Training KMeans Clusterer...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")
kmeans.fit(X_scaled)

# Save the models and preprocessors
models_dir = "/Users/Agentic RAN/models"
os.makedirs(models_dir, exist_ok=True)
joblib.dump(rf_model, os.path.join(models_dir, "rf_model.joblib"))
joblib.dump(kmeans, os.path.join(models_dir, "kmeans_model.joblib"))
joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
joblib.dump(label_enc, os.path.join(models_dir, "label_enc.joblib"))

print(f"Models successfully trained and saved to {models_dir}")
