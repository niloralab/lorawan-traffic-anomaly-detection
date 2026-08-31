from pathlib import Path

import pandas as pd

import evaluate_contamination_sensitivity as experiment


CONTAMINATION_VALUES = [
    0.01,
    0.03,
    0.05,
    0.07,
    0.10,
]

RESULTS_DIR = Path("results")

SUMMARY_OUTPUT_PATH = RESULTS_DIR / (
    "contamination_sensitivity_metrics.csv"
)
def contamination_tag(value):
    return f"{int(round(value * 100)):03d}"
def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_tables = []

    for contamination in CONTAMINATION_VALUES:
        tag = contamination_tag(contamination)

        metrics_path = RESULTS_DIR / (
            f"contamination_{tag}_metrics.csv"
        )

        confusion_matrix_path = RESULTS_DIR / (
            f"contamination_{tag}_confusion_matrix.png"
        )

        print()
        print(
            "Running contamination:",
            contamination,
        )

        experiment.CONTAMINATION = contamination
        experiment.METRICS_OUTPUT_PATH = metrics_path
        experiment.CONFUSION_MATRIX_OUTPUT_PATH = (
            confusion_matrix_path
        )

        experiment.main()

        metrics = pd.read_csv(metrics_path)
        metrics.insert(
            0,
            "contamination",
            contamination,
        )

        result_tables.append(metrics)

    summary = pd.concat(
        result_tables,
        ignore_index=True,
    )

    summary.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )

    print()
    print("Sensitivity-analysis summary:")
    print(
        summary[
            [
                "contamination",
                "true_positives",
                "false_positives",
                "false_negatives",
                "true_negatives",
                "precision",
                "recall",
                "f1_score",
                "false_positive_rate",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        "Summary saved to:",
        SUMMARY_OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
