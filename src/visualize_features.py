from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

INPUT_FILE = "data/processed/features.csv"
RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

sns.set_theme(style="whitegrid")


# 1. Radio and payload distributions
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

sns.histplot(
    data=df,
    x="rssi",
    bins=40,
    kde=True,
    ax=axes[0],
    color="steelblue",
)
axes[0].set_title("RSSI distribution")
axes[0].set_xlabel("RSSI (dBm)")

sns.histplot(
    data=df,
    x="snr",
    bins=40,
    kde=True,
    ax=axes[1],
    color="darkorange",
)
axes[1].set_title("SNR distribution")
axes[1].set_xlabel("SNR (dB)")

sns.histplot(
    data=df,
    x="payload_size_bytes",
    bins=40,
    kde=True,
    ax=axes[2],
    color="seagreen",
)
axes[2].set_title("Payload size distribution")
axes[2].set_xlabel("Payload size (bytes)")

fig.tight_layout()
fig.savefig(
    RESULTS_DIR / "radio_payload_distributions.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close(fig)


# 2. Inter-arrival-time distribution
inter_arrival = df["inter_arrival_time"].dropna()
log_inter_arrival = df["log_inter_arrival_time"].dropna()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

sns.histplot(
    inter_arrival,
    bins=40,
    ax=axes[0],
    color="mediumpurple",
)
axes[0].set_title("Inter-arrival time")
axes[0].set_xlabel("Seconds")

sns.histplot(
    log_inter_arrival,
    bins=40,
    kde=True,
    ax=axes[1],
    color="purple",
)
axes[1].set_title("Log-transformed inter-arrival time")
axes[1].set_xlabel("log(1 + seconds)")

fig.tight_layout()
fig.savefig(
    RESULTS_DIR / "inter_arrival_distributions.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close(fig)


# 3. Temporal-segment lengths
segment_lengths = df.groupby("session_id").size()

segment_categories = pd.cut(
    segment_lengths,
    bins=[0, 1, 2, 3, 5, 10, np.inf],
    labels=["1", "2", "3", "4–5", "6–10", "11+"],
)

segment_summary = (
    segment_categories.value_counts()
    .reindex(["1", "2", "3", "4–5", "6–10", "11+"])
    .reset_index()
)

segment_summary.columns = [
    "Observations per segment",
    "Number of segments",
]

fig, ax = plt.subplots(figsize=(8, 4.5))

sns.barplot(
    data=segment_summary,
    x="Observations per segment",
    y="Number of segments",
    ax=ax,
    color="teal",
)

ax.set_title("Temporal-segment length distribution")
ax.set_xlabel("Observations in a temporal segment")
ax.set_ylabel("Number of temporal segments")

fig.tight_layout()
fig.savefig(
    RESULTS_DIR / "temporal_segment_lengths.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close(fig)


# 4. Feature correlation
correlation_features = [
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

correlation = df[correlation_features].corr()

fig, ax = plt.subplots(figsize=(11, 8))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    ax=ax,
)

ax.set_title("Feature correlation heatmap")

fig.tight_layout()
fig.savefig(
    RESULTS_DIR / "feature_correlation_heatmap.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close(fig)


print("Plots saved to results/:")
print("- radio_payload_distributions.png")
print("- inter_arrival_distributions.png")
print("- temporal_segment_lengths.png")
print("- feature_correlation_heatmap.png")
print()
print("Temporal-segment summary:")
print(segment_summary.to_string(index=False))
