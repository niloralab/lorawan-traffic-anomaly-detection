import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

INPUT_FILE = "data/processed/ml_data.csv"
OUTPUT_FILE = "data/processed/anomaly_results.csv"

MODEL_FEATURES = [
    "rssi",
    "snr",
    "payload_size_bytes",
    "log_inter_arrival_time",
    "rssi_change",
    "snr_change",
    "payload_size_change",
    "counter_reset_or_wrap",
    "retransmission_or_reuse",
    "log_missing_counter_count",
]

# Load the prepared dataset
df = pd.read_csv(INPUT_FILE)

# Select only the numeric features used by the model
X = df[MODEL_FEATURES]

# Scale the features using statistics that are resistant to outliers
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Train an unsupervised anomaly-detection model
model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42,
)

predictions = model.fit_predict(X_scaled)
scores = model.decision_function(X_scaled)

# Isolation Forest returns -1 for anomalies and 1 for normal observations
df["anomaly"] = (predictions == -1).astype(int)

# Lower scores indicate more unusual observations
df["anomaly_score"] = scores

# Sort the most unusual observations first
df = df.sort_values("anomaly_score").reset_index(drop=True)

df.to_csv(OUTPUT_FILE, index=False)

print(f"Input rows: {len(df)}")
print(f"Detected anomalies: {df['anomaly'].sum()}")
print(f"Anomaly percentage: {df['anomaly'].mean() * 100:.2f}%")
print(f"Results saved to: {OUTPUT_FILE}")
print()
print("Most unusual observations:")
print(
    df[
        [
            "event_time",
            "dev_addr",
            "gateway_id",
            "anomaly_score",
            "anomaly",
        ]
    ].head(10).to_string(index=False)
)
