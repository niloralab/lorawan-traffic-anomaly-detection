from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import RobustScaler


INPUT_PATH = Path("data/processed/ml_data.csv")

TRAIN_RATIO = 0.70
SYNTHETIC_ANOMALY_RATIO = 0.10
CONTAMINATION = 0.05
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
    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["event_time"],
    )
    df = df.sort_values("event_time").reset_index(drop=True)

    # Use older observations for training and newer observations for testing
    split_index = int(len(df) * TRAIN_RATIO)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    # Separate predictive features from metadata
    X_train = train_df[MODEL_FEATURES].copy()
    X_test = test_df[MODEL_FEATURES].copy()

    # Initially treat unchanged test observations as baseline observations
    y_test = pd.Series(
        0,
        index=X_test.index,
        name="synthetic_label",
    )

    # Reproducibly select test observations for anomaly injection
    rng = np.random.default_rng(RANDOM_STATE)
    anomaly_count = int(
        len(X_test) * SYNTHETIC_ANOMALY_RATIO
    )

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

    # Learn scaling parameters only from training observations
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply the training transformation to modified test observations
    X_test_scaled = scaler.transform(X_test)

    # Train the model only on historical training observations
    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_scaled)

    # Score and predict future testing observations
    anomaly_scores = model.decision_function(
        X_test_scaled
    )
    raw_predictions = model.predict(X_test_scaled)

    # Convert Isolation Forest output:
    #  1 means normal  -> 0
    # -1 means anomaly -> 1
    y_pred = (raw_predictions == -1).astype(int)

    # Compare predictions with controlled synthetic labels
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    # Combine metadata, modified features, labels, and predictions
    evaluation_df = test_df[
        [
            "event_time",
            "dev_addr",
            "gateway_id",
        ]
    ].copy()

    evaluation_df[MODEL_FEATURES] = X_test[MODEL_FEATURES]
    evaluation_df["synthetic_label"] = y_test
    evaluation_df["prediction"] = y_pred
    evaluation_df["anomaly_score"] = anomaly_scores

    # Find unchanged observations classified as anomalous
    false_positives = evaluation_df[
        (evaluation_df["synthetic_label"] == 0)
        & (evaluation_df["prediction"] == 1)
    ].sort_values(
        "anomaly_score",
        ascending=True,
    )

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
    print(
        "Number of model features:",
        len(MODEL_FEATURES),
    )
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print(
        "X_train_scaled shape:",
        X_train_scaled.shape,
    )
    print(
        "X_test_scaled shape:",
        X_test_scaled.shape,
    )

    # Report the synthetic labels
    print()
    print(
        "Injected synthetic anomalies:",
        int(y_test.sum()),
    )
    print(
        "Synthetic anomaly percentage:",
        f"{y_test.mean() * 100:.2f}%",
    )

    print()
    print("Label distribution:")
    print(y_test.value_counts().sort_index())

    # Report model evaluation results
    print()
    print("Evaluation results:")
    print("True negatives:", tn)
    print("False positives:", fp)
    print("False negatives:", fn)
    print("True positives:", tp)

    print()
    print("Precision:", f"{precision:.3f}")
    print("Recall:", f"{recall:.3f}")
    print("F1-score:", f"{f1:.3f}")
    print(
        "Predicted anomalies:",
        int(y_pred.sum()),
    )

    # Display the most unusual unchanged observations
    print()
    print(
        "Most unusual unchanged test observations:"
    )

    inspection_columns = [
        "event_time",
        "dev_addr",
        "gateway_id",
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
        "anomaly_score",
    ]

    print(
        false_positives[inspection_columns]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
