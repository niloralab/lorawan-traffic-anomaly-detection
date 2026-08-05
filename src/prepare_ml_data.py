import numpy as np
import pandas as pd

INPUT_FILE = "data/processed/features.csv"
OUTPUT_FILE = "data/processed/ml_data.csv"

MODEL_FEATURES = [
    "rssi",
    "snr",
    "payload_size_bytes",
    "log_inter_arrival_time",
    "payload_size_change",
    "counter_reset_or_wrap",
    "retransmission_or_reuse",
    "log_missing_counter_count",
    "gateway_changed",
    "log_counter_decrease_magnitude",
]

METADATA_COLUMNS = [
    "event_time",
    "source_file",
    "dev_addr",
    "session_id",
    "gateway_id",
]

df = pd.read_csv(INPUT_FILE)

selected_columns = METADATA_COLUMNS + MODEL_FEATURES
ml_data = df[selected_columns].copy()

# Replace infinite values with missing values
ml_data = ml_data.replace([np.inf, -np.inf], np.nan)

rows_before_cleaning = len(ml_data)

# Keep only observations with all modelling features available
ml_data = ml_data.dropna(subset=MODEL_FEATURES).copy()
ml_data = ml_data.reset_index(drop=True)

ml_data.to_csv(OUTPUT_FILE, index=False)

print(f"Input feature rows: {rows_before_cleaning}")
print(f"Rows usable for ML: {len(ml_data)}")
print(f"Rows removed: {rows_before_cleaning - len(ml_data)}")
print(f"ML data saved to: {OUTPUT_FILE}")
print()
print("Selected modelling features:")
for feature in MODEL_FEATURES:
    print(f"- {feature}")
