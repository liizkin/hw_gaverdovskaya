import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = BASE_DIR / "train.csv"
TARGET = "price_range"
EXPECTED_CLASSES = [0, 1, 2, 3]
FEATURES = [
    "battery_power",
    "blue",
    "clock_speed",
    "dual_sim",
    "fc",
    "four_g",
    "int_memory",
    "m_dep",
    "mobile_wt",
    "n_cores",
    "pc",
    "px_height",
    "px_width",
    "ram",
    "sc_h",
    "sc_w",
    "talk_time",
    "three_g",
    "touch_screen",
    "wifi",
]


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def format_number(value):
    return f"{value:.4f}"


df = pd.read_csv(CSV_PATH)

if TARGET not in df.columns:
    raise ValueError(f"The required target column {TARGET!r} is missing.")

missing_columns = sorted(set(FEATURES + [TARGET]) - set(df.columns))
if missing_columns:
    raise ValueError(f"Required columns are missing: {missing_columns}")

shape = [int(df.shape[0]), int(df.shape[1])]
dtypes = {column: str(dtype) for column, dtype in df.dtypes.items()}
missing_by_column = {
    column: int(count) for column, count in df.isna().sum().items()
}
duplicate_rows = int(df.duplicated().sum())

class_counts = {
    str(label): int((df[TARGET] == label).sum()) for label in EXPECTED_CLASSES
}
expected_per_class = len(df) / len(EXPECTED_CLASSES) if EXPECTED_CLASSES else 0
balance_tolerance = expected_per_class * 0.10
price_range_balanced = bool(
    expected_per_class > 0
    and all(
        abs(class_counts[str(label)] - expected_per_class) <= balance_tolerance
        for label in EXPECTED_CLASSES
    )
)

resolution_zero_counts = {
    column: int((df[column] == 0).sum())
    for column in ("px_height", "px_width")
}

granularity = {}
for column in ("fc", "pc", "clock_speed"):
    values = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy()
    unique_values = np.sort(np.unique(values))
    half_step_count = int(np.isclose(np.mod(values, 1), 0.5).sum())
    granularity[column] = {
        "unique_value_count": int(unique_values.size),
        "integer_valued": bool(np.isclose(values, np.round(values)).all()),
        "half_step_observation_count": half_step_count,
        "unique_values": [
            int(value) if np.isclose(value, round(value)) else float(value)
            for value in unique_values
        ],
    }

constant_features = [
    column for column in df.columns if int(df[column].nunique(dropna=False)) <= 1
]

overall_describe = df.describe(include="all").transpose()
overall_describe.to_csv(OUTPUT_DIR / "descriptive_stats.csv", index_label="feature")

classwise_describe = {}
classwise_rows = []
for feature in FEATURES:
    classwise_describe[feature] = {}
    for label in EXPECTED_CLASSES:
        values = df.loc[df[TARGET] == label, feature]
        statistics = values.describe().to_dict()
        classwise_describe[feature][str(label)] = statistics
        classwise_rows.append(
            {"feature": feature, "price_range": label, **statistics}
        )

pd.DataFrame(classwise_rows).to_csv(
    OUTPUT_DIR / "descriptive_stats_by_price_range.csv", index=False
)

correlation_matrix = df.corr(numeric_only=True)
target_correlations = correlation_matrix[TARGET].drop(labels=[TARGET])
top_features = target_correlations.abs().sort_values(ascending=False).head(5)

fig, ax = plt.subplots(figsize=(12, 8), dpi=120)
sns.heatmap(
    correlation_matrix,
    cmap="vlag",
    center=0,
    linewidths=0.25,
    cbar_kws={"shrink": 0.8},
    ax=ax,
)
ax.set_title("Feature Correlations")
plt.tight_layout()
corr_path = OUTPUT_DIR / "corr.png"
fig.savefig(corr_path, dpi=120, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(12, 8), dpi=120)
for ax, feature in zip(axes, ("ram", "battery_power", "px_height")):
    sns.boxplot(
        data=df,
        x=TARGET,
        y=feature,
        order=EXPECTED_CLASSES,
        color="#2A788E",
        ax=ax,
    )
    ax.set_title(feature.replace("_", " ").title())
    ax.set_xlabel("Price range")
    ax.set_ylabel(feature.replace("_", " ").title())
plt.tight_layout()
boxplots_path = OUTPUT_DIR / "boxplots.png"
fig.savefig(boxplots_path, dpi=120, bbox_inches="tight")
plt.close(fig)

for figure_path in (corr_path, boxplots_path):
    if figure_path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError(f"Generated figure exceeds 2 MB: {figure_path}")

anomalies = []
if not price_range_balanced:
    anomalies.append(
        "Price classes are imbalanced: observed counts are "
        f"{class_counts}, versus about {expected_per_class:.1f} rows per class."
    )
for column, count in resolution_zero_counts.items():
    if count:
        anomalies.append(
            f"{column} contains {count} zero value(s), which are suspicious because "
            "a real phone cannot have zero pixel resolution."
        )
if duplicate_rows:
    anomalies.append(f"The dataset contains {duplicate_rows} duplicate row(s).")
nonzero_missing = {
    column: count for column, count in missing_by_column.items() if count
}
if nonzero_missing:
    anomalies.append(f"Missing values were found: {nonzero_missing}.")
if constant_features:
    anomalies.append(
        "Constant feature(s) with zero variance were found: "
        f"{constant_features}."
    )

report_lines = [
    "# Mobile Phone Price EDA",
    "",
    f"The dataset contains {shape[0]} rows and {shape[1]} columns; "
    f"price classes are {'balanced' if price_range_balanced else 'not balanced'} "
    "and the associations below are correlations, not evidence of causation.",
    "",
    "## Five strongest feature associations",
    "",
]

business_reasoning = {
    "ram": "RAM is the clearest basis for price-tier segmentation because devices with more working memory are strongly concentrated in higher tiers.",
    "battery_power": "Battery capacity is a useful secondary differentiator because longer-lasting devices tend to occupy higher price tiers.",
    "px_width": "Display width resolution can support premium positioning because sharper displays tend to appear in more expensive tiers.",
    "px_height": "Display height resolution can support premium positioning because sharper displays tend to appear in more expensive tiers.",
    "int_memory": "Internal storage is a modest differentiator because greater built-in capacity is associated with somewhat higher price tiers.",
}

for feature in top_features.index:
    coefficient = float(target_correlations[feature])
    rationale = business_reasoning.get(
        feature,
        "This feature can help refine price tiers because its measured values vary with the ordinal price segment.",
    )
    report_lines.append(
        f"- **{feature}** (Pearson r = {format_number(coefficient)}): {rationale}"
    )

report_lines.extend(["", "## Data-quality findings", ""])
if not anomalies:
    report_lines.extend(
        [
            "No class imbalance, zero-resolution readings, duplicates, missing values, or constant features were found.",
            "",
        ]
    )
else:
    for anomaly in anomalies:
        report_lines.extend([anomaly, ""])

for column, details in granularity.items():
    if column in ("fc", "pc"):
        value_type = "integer-valued" if details["integer_valued"] else "non-integer"
        observation = (
            f"{column} has {details['unique_value_count']} {value_type} levels and "
            "is a discrete count, not a continuous measurement."
        )
    else:
        half_step_values = [
            value for value in details["unique_values"]
            if np.isclose(np.mod(value, 1), 0.5)
        ]
        examples = ", ".join(str(value) for value in half_step_values[:6])
        observation = (
            f"{column} has {details['unique_value_count']} recorded levels and is "
            f"discrete rather than truly continuous; {details['half_step_observation_count']} "
            f"observations are half-steps (including {examples})."
        )
    report_lines.extend([f"Granularity check: {observation}", ""])

(BASE_DIR / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

summary = {
    "input_file": CSV_PATH.name,
    "shape": {"rows": shape[0], "columns": shape[1]},
    "dtypes": dtypes,
    "missing_values_by_column": missing_by_column,
    "missing_value_total": int(sum(missing_by_column.values())),
    "duplicate_rows": duplicate_rows,
    "describe": overall_describe.to_dict(orient="index"),
    "describe_by_price_range": classwise_describe,
    "price_range_balance": {
        "balanced": price_range_balanced,
        "class_counts": class_counts,
        "expected_rows_per_class": expected_per_class,
        "tolerance_percent": 10,
    },
    "resolution_zero_counts": resolution_zero_counts,
    "resolution_zeros_suspicious": {
        column: bool(count > 0) for column, count in resolution_zero_counts.items()
    },
    "granularity_checks": granularity,
    "constant_features": constant_features,
    "top_5_absolute_pearson_correlations_with_price_range": {
        feature: float(target_correlations[feature]) for feature in top_features.index
    },
    "anomalies": anomalies,
    "figure_files": {
        "correlation_heatmap": str(corr_path.relative_to(BASE_DIR)),
        "feature_boxplots": str(boxplots_path.relative_to(BASE_DIR)),
    },
    "figure_size_inches": [12, 8],
    "figure_dpi": 120,
    "figure_size_bytes": {
        corr_path.name: corr_path.stat().st_size,
        boxplots_path.name: boxplots_path.stat().st_size,
    },
}

with (OUTPUT_DIR / "eda_summary.json").open("w", encoding="utf-8") as summary_file:
    json.dump(json_safe(summary), summary_file, indent=2, allow_nan=False)

print(f"Shape: {shape}")
print(f"Dtypes: {dtypes}")
print(f"Missing values by column: {missing_by_column}")
print(f"Duplicate rows: {duplicate_rows}")
print("Overall describe():")
print(overall_describe.to_string())
print("Describe() by price_range:")
print(pd.DataFrame(classwise_rows).to_string(index=False))
print("Top feature correlations:")
print(top_features.map(format_number).to_string())
print(f"Wrote {OUTPUT_DIR} and {BASE_DIR / 'report.md'}")