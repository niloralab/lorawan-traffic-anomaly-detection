from pathlib import Path

import matplotlib.pyplot as plt
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

METRICS_OUTPUT_PATH = Path(
    "results/contamination_010_metrics.csv"
)
CONFUSION_MATRIX_OUTPUT_PATH = Path(
    "results/contamination_010_confusion_matrix.png"
)

TRAIN_RATIO = 0.70
SYNTHETIC_ANOMALY_RATIO = 0.10
CONTAMINATION = 0.10
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


def save_confusion_matrix(tn, fp, fn, tp):
    matrix = np.array(
        [
            [tn, fp],
            [fn, tp],
        ]
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(
        matrix,
        cmap="Blues",
    )

    ax.set_title(
        "Synthetic Anomaly Detection\nConfusion Matrix"
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Synthetic label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticklabels(["Normal", "Anomaly"])

    threshold = matrix.max() / 2

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]

            ax.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > threshold
                    else "black"
                ),
                fontsize=13,
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    fig.savefig(
        CONFUSION_MATRIX_OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    # Create the results directory if it does not exist
    METRICS_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load and chronologically order the observations
    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["event_time"],
    )

    df = df.sort_values(
        "event_time"
    ).reset_index(drop=True)

    # Use older observations for training
    split_index = int(len(df) * TRAIN_RATIO)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    # Separate model features from metadata
    X_train = train_df[MODEL_FEATURES].copy()
    X_test = test_df[MODEL_FEATURES].copy()

    # Initially label test observations as unchanged
    y_test = pd.Series(
        0,
        index=X_test.index,
        name="synthetic_label",
    )

    # Select reproducible observations for injection
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

    # Label modified observations as synthetic anomalies
    y_test.loc[anomaly_indices] = 1

    # Learn scaling only from training observations
    scaler = RobustScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # Train Isolation Forest on historical observations
    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train_scaled)

    # Score and classify testing observations
    anomaly_scores = model.decision_function(
        X_test_scaled
    )

    raw_predictions = model.predict(
        X_test_scaled
    )

    # Isolation Forest:
    #  1 = normal  -> 0
    # -1 = anomaly -> 1
    y_pred = (
        raw_predictions == -1
    ).astype(int)

    # Calculate the confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    ).ravel()

    # Calculate evaluation metrics
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

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    # Combine metadata, features, and predictions
    evaluation_df = test_df[
        [
            "event_time",
            "dev_addr",
            "gateway_id",
        ]
    ].copy()

    evaluation_df[MODEL_FEATURES] = (
        X_test[MODEL_FEATURES]
    )

    evaluation_df["synthetic_label"] = y_test
    evaluation_df["prediction"] = y_pred
    evaluation_df["anomaly_score"] = (
        anomaly_scores
    )

    # Find unchanged observations flagged as anomalous
    false_positives = evaluation_df[
        (evaluation_df["synthetic_label"] == 0)
        & (evaluation_df["prediction"] == 1)
    ].sort_values(
        "anomaly_score",
        ascending=True,
    )

    # Save evaluation metrics
    metrics_df = pd.DataFrame(
        [
            {
                "total_observations": len(df),
                "training_observations": len(
                    train_df
                ),
                "testing_observations": len(
                    test_df
                ),
                "synthetic_anomalies": int(
                    y_test.sum()
                ),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "false_positive_rate": (
                    false_positive_rate
                ),
            }
        ]
    )

    metrics_df.to_csv(
        METRICS_OUTPUT_PATH,
        index=False,
    )

    # Save the confusion-matrix figure
    save_confusion_matrix(
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
    )

    # Report dataset split
    print("Total observations:", len(df))
    print(
        "Training observations:",
        len(train_df),
    )
    print(
        "Testing observations:",
        len(test_df),
    )

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

    # Report model dimensions
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

    # Report synthetic labels
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
    print(
        y_test.value_counts().sort_index()
    )

    # Report evaluation results
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
        "False-positive rate:",
        f"{false_positive_rate:.3f}",
    )
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

    print()
    print(
        "Saved evaluation metrics to:",
        METRICS_OUTPUT_PATH,
    )
    print(
        "Saved confusion matrix to:",
        CONFUSION_MATRIX_OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
