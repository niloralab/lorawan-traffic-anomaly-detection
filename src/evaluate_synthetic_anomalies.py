from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


INPUT_PATH = Path("data/processed/ml_data.csv")
TRAIN_RATIO = 0.70
SYNTHETIC_ANOMALY_RATIO = 0.10
RANDOM_STATE = 42

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

def main():
    # Load and chronologically order the modelling observations
    df = pd.read_csv(INPUT_PATH, parse_dates=["event_time"])
    df = df.sort_values("event_time").reset_index(drop=True)

    # Use older observations for training and newer observations for testing
    split_index = int(len(df) * TRAIN_RATIO)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    # Separate predictive features from metadata
    X_train = train_df[MODEL_FEATURES].copy()
    X_test = test_df[MODEL_FEATURES].copy()

    # Initially treat unchanged test observations as baseline observations
    y_test = pd.Series(0, index=X_test.index, name="synthetic_label")

    # Reproducibly select test observations for anomaly injection
    rng = np.random.default_rng(RANDOM_STATE)
    anomaly_count = int(len(X_test) * SYNTHETIC_ANOMALY_RATIO)

    anomaly_indices = rng.choice(
        X_test.index,
        size=anomaly_count,
        replace=False,
    )

    # Inject controlled synthetic anomalies
    X_test.loc[
        anomaly_indices,
        "log_inter_arrival_time",
    ] += np.log(20)

    X_test.loc[
        anomaly_indices,
        "payload_size_change",
    ] += 50

    X_test.loc[
        anomaly_indices,
        "counter_reset_or_wrap",
    ] = 1

    X_test.loc[
        anomaly_indices,
        "log_counter_decrease_magnitude",
    ] += np.log1p(5000)

    # Record which test observations were modified
    y_test.loc[anomaly_indices] = 1

    # Learn scaling parameters only from the training observations
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply the training transformation to the modified test observations
    X_test_scaled = scaler.transform(X_test)

    # Report the temporal split
    print("Total observations:", len(df))
    print("Training observations:", len(train_df))
    print("Testing observations:", len(test_df))

    print()
    print("Training period:")
    print(
        train_df["event_time"].min(),
        "to",
        train_df["event_time"].max(),
    )

    print()
    print("Testing period:")
    print(
        test_df["event_time"].min(),
        "to",
        test_df["event_time"].max(),
    )

    # Report the modelling matrix dimensions
    print()
    print("Number of model features:", len(MODEL_FEATURES))
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("X_train_scaled shape:", X_train_scaled.shape)
    print("X_test_scaled shape:", X_test_scaled.shape)

    # Report the synthetic labels
    print()
    print("Injected synthetic anomalies:", int(y_test.sum()))
    print(
        "Synthetic anomaly percentage:",
        f"{y_test.mean() * 100:.2f}%",
    )

    print()
    print("Label distribution:")
    print(y_test.value_counts().sort_index())

if __name__ == "__main__":
    main()
