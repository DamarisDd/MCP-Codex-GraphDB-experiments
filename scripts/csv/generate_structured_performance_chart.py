#!/usr/bin/env python3
"""Generate the run-comparison figure from evaluation_all_per_run.csv.

The script calculates macro-averaged precision, recall and F1 only over
structured-answer runs. Binary answers are deliberately excluded because they
are evaluated by accuracy rather than by set-based precision and recall.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


SCRIPT_DIR = Path(__file__).resolve().parent

REQUIRED_COLUMNS = {
    "query_id",
    "run_id",
    "response_mode",
    "precision",
    "recall",
    "f1",
    "exact_answer",
}

COLORS = {
    "Precision": "#4C78A8",
    "Recall": "#16858C",
    "F1": "#D59A2A",
    "Exact answers": "#D66A5E",
}

MARKERS = {
    "Precision": "o",
    "Recall": "s",
    "F1": "D",
    "Exact answers": "^",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate run-level macro-averages for structured answers and "
            "export the comparison figure as PNG and PDF."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=SCRIPT_DIR / "evaluation_all_per_run.csv",
        help="Path to evaluation_all_per_run.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "figures",
        help="Directory in which the PNG and PDF files will be written.",
    )
    parser.add_argument(
        "--prefix",
        default="structured_performance_across_runs",
        help="Output filename without an extension.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=320,
        help="PNG resolution in dots per inch (default: 320).",
    )
    return parser.parse_args()


def parse_boolean(value: str, *, field: str, row_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(
        f"Invalid Boolean value {value!r} in column {field!r}, CSV row {row_number}."
    )


def read_structured_rows(csv_path: Path) -> list[dict[str, object]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path.resolve()}")

    structured_rows: list[dict[str, object]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        available_columns = set(reader.fieldnames or [])
        missing_columns = REQUIRED_COLUMNS - available_columns
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"The CSV is missing required columns: {missing}")

        for row_number, row in enumerate(reader, start=2):
            if row["response_mode"].strip().lower() != "structured":
                continue

            try:
                run_id = int(row["run_id"])
                precision = float(row["precision"])
                recall = float(row["recall"])
                f1 = float(row["f1"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid numeric value in CSV row {row_number}: {exc}"
                ) from exc

            for metric_name, metric_value in {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }.items():
                if not 0.0 <= metric_value <= 1.0:
                    raise ValueError(
                        f"{metric_name} must be between 0 and 1 in CSV row {row_number}."
                    )

            query_id = row["query_id"].strip()
            if not query_id:
                raise ValueError(f"Missing query_id in CSV row {row_number}.")

            structured_rows.append(
                {
                    "query_id": query_id,
                    "run_id": run_id,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "exact_answer": parse_boolean(
                        row["exact_answer"],
                        field="exact_answer",
                        row_number=row_number,
                    ),
                }
            )

    if not structured_rows:
        raise ValueError("The CSV contains no rows with response_mode='structured'.")
    return structured_rows


def calculate_run_metrics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_run: dict[int, dict[str, dict[str, object]]] = defaultdict(dict)

    for row in rows:
        run_id = int(row["run_id"])
        query_id = str(row["query_id"])
        if query_id in by_run[run_id]:
            raise ValueError(f"Duplicate structured row for {query_id}, Run {run_id}.")
        by_run[run_id][query_id] = row

    run_ids = sorted(by_run)
    reference_run = run_ids[0]
    reference_queries = set(by_run[reference_run])

    for run_id in run_ids[1:]:
        current_queries = set(by_run[run_id])
        if current_queries != reference_queries:
            missing = sorted(reference_queries - current_queries)
            additional = sorted(current_queries - reference_queries)
            raise ValueError(
                "Structured-query sets differ between runs. "
                f"Run {run_id} missing={missing}, additional={additional}."
            )

    metrics: list[dict[str, object]] = []
    for run_id in run_ids:
        run_rows = list(by_run[run_id].values())
        exact_count = sum(bool(row["exact_answer"]) for row in run_rows)
        metrics.append(
            {
                "run_id": run_id,
                "query_count": len(run_rows),
                "precision": statistics.fmean(float(row["precision"]) for row in run_rows),
                "recall": statistics.fmean(float(row["recall"]) for row in run_rows),
                "f1": statistics.fmean(float(row["f1"]) for row in run_rows),
                "exact_count": exact_count,
                "exact_rate": exact_count / len(run_rows),
            }
        )

    return metrics


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 8.5,
            "axes.edgecolor": "#9AA9B5",
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def label_offsets(series_name: str, run_count: int) -> list[float]:
    # Tuned for the three-run experiment; sensible defaults are used otherwise.
    if run_count == 3:
        return {
            "Precision": [1.2, 1.1, 1.3],
            "Recall": [0.9, 0.9, 0.6],
            "F1": [-1.0, -1.0, -1.1],
            "Exact answers": [-1.25, -1.25, -1.25],
        }[series_name]
    default = {
        "Precision": 1.2,
        "Recall": 0.8,
        "F1": -1.0,
        "Exact answers": -1.25,
    }[series_name]
    return [default] * run_count


def create_figure(metrics: list[dict[str, object]], output_dir: Path, prefix: str, dpi: int) -> tuple[Path, Path]:
    configure_style()

    labels = [f"Run {row['run_id']}" for row in metrics]
    x_values = list(range(len(metrics)))
    series = {
        "Precision": [100 * float(row["precision"]) for row in metrics],
        "Recall": [100 * float(row["recall"]) for row in metrics],
        "F1": [100 * float(row["f1"]) for row in metrics],
        "Exact answers": [100 * float(row["exact_rate"]) for row in metrics],
    }

    all_values = [value for values in series.values() for value in values]
    lower_limit = max(0, 5 * math.floor((min(all_values) - 5) / 5))

    fig, ax = plt.subplots(figsize=(7.4, 4.45))
    for series_name, values in series.items():
        ax.plot(
            x_values,
            values,
            label=series_name,
            color=COLORS[series_name],
            marker=MARKERS[series_name],
            linewidth=2,
            markersize=6,
        )

        offsets = label_offsets(series_name, len(metrics))
        for index, (value, offset) in enumerate(zip(values, offsets)):
            vertical_alignment = "bottom" if offset > 0 else "top"
            point_label = f"{value:.1f}%"
            if series_name == "Exact answers":
                point_label += (
                    f" ({metrics[index]['exact_count']}/{metrics[index]['query_count']})"
                )
            ax.text(
                index,
                value + offset,
                point_label,
                ha="center",
                va=vertical_alignment,
                fontsize=7.7,
                color=COLORS[series_name],
            )

    run_count = len(metrics)
    title = (
        " "
        if run_count == 3
        else " "
    )
    query_count = int(metrics[0]["query_count"])
    ax.set_title(title, loc="left", color="#17324D", pad=14)
    ax.text(
        0,
        1.015,
        f" ",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.6,
        color="#526170",
    )
    ax.set_xlabel("Run position")
    ax.set_ylabel("Mean score")
    ax.set_xticks(x_values, labels)
    ax.set_xlim(-0.18, len(metrics) - 0.82)
    ax.set_ylim(lower_limit, 100)
    ax.set_yticks(list(range(int(lower_limit), 101, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
    ax.grid(axis="y", color="#D7E0E7", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.29))

    fig.text(
        0.99,
        0.012,
        " ",
        ha="right",
        va="bottom",
        fontsize=7.6,
        color="#64748B",
    )
    fig.subplots_adjust(left=0.10, right=0.985, top=0.84, bottom=0.27)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{prefix}.png"
    pdf_path = output_dir / f"{prefix}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def print_summary(metrics: list[dict[str, object]]) -> None:
    print("\nRun-level structured-answer metrics")
    print("Run  Precision  Recall  F1      Exact")
    for row in metrics:
        print(
            f"{row['run_id']:>3}  "
            f"{100 * float(row['precision']):>8.1f}%  "
            f"{100 * float(row['recall']):>6.1f}%  "
            f"{100 * float(row['f1']):>5.1f}%  "
            f"{row['exact_count']}/{row['query_count']} "
            f"({100 * float(row['exact_rate']):.1f}%)"
        )


def main() -> None:
    args = parse_arguments()
    rows = read_structured_rows(args.csv)
    metrics = calculate_run_metrics(rows)
    png_path, pdf_path = create_figure(
        metrics=metrics,
        output_dir=args.output_dir,
        prefix=args.prefix,
        dpi=args.dpi,
    )
    print_summary(metrics)
    print(f"\nPNG written to: {png_path.resolve()}")
    print(f"PDF written to: {pdf_path.resolve()}")


if __name__ == "__main__":
    main()
