import numpy as np
import pandas as pd

INPUT_FILE = "data/processed/uplinks.csv"
OUTPUT_FILE = "data/processed/features.csv"

df = pd.read_csv(INPUT_FILE)

# Convert time to datetime
df["event_time"] = pd.to_datetime(
    df["event_time"],
    errors="coerce",
    utc=True,
)

# Keep ordinary data uplinks
df = df[df["is_join_request"] == False].copy()

# DevAddr is a session-level identifier, not a permanent device identifier
df = df.dropna(subset=["event_time", "dev_addr"])
df = df.sort_values(["dev_addr", "event_time"])

grouped = df.groupby("dev_addr", group_keys=False)

# Time between consecutive observations
df["inter_arrival_time"] = (
    grouped["event_time"].diff().dt.total_seconds()
)

# Log transformation reduces the effect of extremely large time gaps
df["log_inter_arrival_time"] = np.log1p(
    df["inter_arrival_time"]
)

# Changes between consecutive observations
df["rssi_change"] = grouped["rssi"].diff()
df["snr_change"] = grouped["snr"].diff()
df["payload_size_change"] = grouped["payload_size_bytes"].diff()
df["f_cnt_gap"] = grouped["f_cnt"].diff()

# FCnt behaviour
df["counter_reset_or_wrap"] = (
    df["f_cnt_gap"] < 0
).astype(int)

df["retransmission_or_reuse"] = (
    df["f_cnt_gap"] == 0
).astype(int)

# Number of counters skipped in a forward sequence
df["missing_counter_count"] = (
    df["f_cnt_gap"] - 1
).clip(lower=0)

df["log_missing_counter_count"] = np.log1p(
    df["missing_counter_count"]
)

# This indicates missing counters, not necessarily actual radio packet loss
df["possible_packet_loss"] = (
    df["missing_counter_count"] > 0
).astype(int)

feature_columns = [
    "event_time",
    "source_file",
    "dev_addr",
    "gateway_id",
    "rssi",
    "snr",
    "payload_size_bytes",
    "f_cnt",
    "inter_arrival_time",
    "log_inter_arrival_time",
    "rssi_change",
    "snr_change",
    "payload_size_change",
    "f_cnt_gap",
    "counter_reset_or_wrap",
    "retransmission_or_reuse",
    "missing_counter_count",
    "log_missing_counter_count",
    "possible_packet_loss",
]

features = df[feature_columns].copy()
features = features.reset_index(drop=True)

features.to_csv(OUTPUT_FILE, index=False)

print(f"Input rows: {len(df)}")
print(f"Features saved to: {OUTPUT_FILE}")
print()
print("Feature availability:")
print(features.notna().sum())
