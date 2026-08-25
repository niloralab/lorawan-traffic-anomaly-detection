import pandas as pd


INPUT_FILE = "data/processed/anomaly_results.csv"
OUTPUT_FILE = "data/processed/analysed_anomalies.csv"
SUMMARY_FILE = "results/anomaly_reason_counts.csv"

# Load the results produced by the Isolation Forest model
df = pd.read_csv(INPUT_FILE)

# Separate observations marked as normal and anomalous
normal = df[df["anomaly"] == 0].copy()
anomalies = df[df["anomaly"] == 1].copy()

# Use the 95th percentile of normal observations as a reference
long_timing_threshold = normal[
    "log_inter_arrival_time"
].quantile(0.95)

large_payload_change_threshold = normal[
    "payload_size_change"
].abs().quantile(0.95)

large_missing_counter_threshold = normal[
    "log_missing_counter_count"
].quantile(0.95)


def identify_possible_reasons(row):
    """
    Describe observable indicators associated with an anomaly.

    These indicators are not confirmed attacks or root causes.
    """

    reasons = []

    if row["counter_reset_or_wrap"] == 1:
        reasons.append("counter_decrease")

    if row["retransmission_or_reuse"] == 1:
        reasons.append("counter_reuse")

    if row["gateway_changed"] == 1:
        reasons.append("gateway_change")

    if row["log_inter_arrival_time"] > long_timing_threshold:
        reasons.append("long_timing")

    if (
        abs(row["payload_size_change"])
        > large_payload_change_threshold
    ):
        reasons.append("large_payload_change")

    if (
        row["log_missing_counter_count"]
        > large_missing_counter_threshold
    ):
        reasons.append("large_missing_counter_gap")

    if not reasons:
        reasons.append("other_multivariate_outlier")

    return "; ".join(reasons)


# Assign one or more observable indicators to each anomaly
anomalies["possible_reasons"] = anomalies.apply(
    identify_possible_reasons,
    axis=1,
)

# Count each indicator separately
reason_counts = (
    anomalies["possible_reasons"]
    .str.split("; ")
    .explode()
    .value_counts()
)

# Place the most unusual observations first
anomalies = anomalies.sort_values("anomaly_score")

# Convert the indicator counts into a table
reason_summary = (
    reason_counts
    .rename_axis("possible_reason")
    .reset_index(name="count")
)

# Save the detailed and summary results
anomalies.to_csv(
    OUTPUT_FILE,
    index=False,
)

reason_summary.to_csv(
    SUMMARY_FILE,
    index=False,
)

# Display the main results
print("Detected anomalies:", len(anomalies))
print("Long timing threshold:", long_timing_threshold)
print(
    "Large payload change threshold:",
    large_payload_change_threshold,
)
print(
    "Large missing-counter threshold:",
    large_missing_counter_threshold,
)

print("\nPossible reason counts:")
print(reason_counts)

print("\nSaved analysed anomalies to:", OUTPUT_FILE)
print("Saved reason summary to:", SUMMARY_FILE)
