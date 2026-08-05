import numpy as np
import pandas as pd

INPUT_FILE = "data/processed/uplinks.csv"
OUTPUT_FILE = "data/processed/features.csv"

# A new approximate session starts after 12 hours of inactivity
SESSION_GAP_SECONDS = 12 * 60 * 60
MAX_VALID_RSSI = -20

# Load parsed uplink events
df = pd.read_csv(INPUT_FILE)

# Convert event time to UTC datetime
df["event_time"] = pd.to_datetime(
    df["event_time"],
    errors="coerce",
    utc=True,
)

# Keep ordinary data uplinks
df = df[df["is_join_request"] == False].copy()
# Mark implausibly high RSSI measurements as missing
invalid_rssi = df["rssi"] > MAX_VALID_RSSI
invalid_rssi_count = invalid_rssi.sum()

df.loc[invalid_rssi, "rssi"] = np.nan

# Remove rows that cannot be assigned to a temporal group
df = df.dropna(subset=["event_time", "dev_addr"])

# DevAddr is not treated as a permanent device identifier
df = df.sort_values(["dev_addr", "event_time"])

devaddr_grouped = df.groupby(
    "dev_addr",
    group_keys=False,
)

# Time since the previous observation with the same DevAddr
df["devaddr_time_gap"] = (
    devaddr_grouped["event_time"]
    .diff()
    .dt.total_seconds()
)

# Start a new approximate session for the first observation or
# after more than 12 hours of inactivity
df["new_session"] = (
    df["devaddr_time_gap"].isna()
    | (df["devaddr_time_gap"] > SESSION_GAP_SECONDS)
)

# Number sessions separately for each DevAddr
df["session_number"] = (
    df.groupby("dev_addr")["new_session"]
    .cumsum()
    .astype(int)
)

# Create an approximate session identifier
df["session_id"] = (
    df["dev_addr"].astype(str)
    + "_session_"
    + df["session_number"].astype(str)
)

# Calculate behavioural changes only within the same session
grouped = df.groupby(
    "session_id",
    group_keys=False,
)

# Identify whether the gateway changed between consecutive observations
df["previous_gateway_id"] = grouped["gateway_id"].shift(1)

df["gateway_changed"] = (
    df["previous_gateway_id"].notna()
    & (df["gateway_id"] != df["previous_gateway_id"])
).astype(int)

# Time between consecutive observations
df["inter_arrival_time"] = (
    grouped["event_time"]
    .diff()
    .dt.total_seconds()
)

# Reduce the influence of extremely large time gaps
df["log_inter_arrival_time"] = np.log1p(
    df["inter_arrival_time"]
)

# Changes between consecutive observations
df["rssi_change"] = grouped["rssi"].diff()
df["snr_change"] = grouped["snr"].diff()
df["payload_size_change"] = grouped["payload_size_bytes"].diff()
df["f_cnt_gap"] = grouped["f_cnt"].diff()

# Indicate a counter decrease inside the same approximate session
df["counter_reset_or_wrap"] = (
    df["f_cnt_gap"] < 0
).astype(int)

# Measure the magnitude of a frame-counter decrease
df["counter_decrease_magnitude"] = (
    -df["f_cnt_gap"]
).clip(lower=0)

# Reduce the influence of extremely large counter decreases
df["log_counter_decrease_magnitude"] = np.log1p(
    df["counter_decrease_magnitude"]
)

# Indicate an unchanged frame counter
df["retransmission_or_reuse"] = (
    df["f_cnt_gap"] == 0
).astype(int)

# Count skipped frame-counter values in a forward sequence
df["missing_counter_count"] = (
    df["f_cnt_gap"] - 1
).clip(lower=0)

# Reduce the influence of very large frame-counter gaps
df["log_missing_counter_count"] = np.log1p(
    df["missing_counter_count"]
)

# A counter gap is only a possible indication of packet loss
df["possible_packet_loss"] = (
    df["missing_counter_count"] > 0
).astype(int)

feature_columns = [
    "event_time",
    "source_file",
    "dev_addr",
    "session_id",
    "session_number",
    "gateway_id",
    "previous_gateway_id",
    "gateway_changed",
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
    "counter_decrease_magnitude",
    "log_counter_decrease_magnitude",
]

features = df[feature_columns].copy()
features = features.reset_index(drop=True)

features.to_csv(OUTPUT_FILE, index=False)

print(f"Input rows: {len(df)}")
print(f"Approximate sessions: {df['session_id'].nunique()}")
print(f"Features saved to: {OUTPUT_FILE}")
print()
print("Feature availability:")
print(features.notna().sum())
print(f"Invalid RSSI measurements: {invalid_rssi_count}")
