import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc


DATA_PATH = "data.csv"

SENSORS = [
    "TGS2600",
    "TGS2602",
    "TGS822",
    "MQ3",
    "MQ135",
    "MQ138",
    "MiCS_NO2",
    "MiCS_NH3",
    "MiCS_CO"
]


print("=" * 70)
print("OLFACT-AI SENSOR THRESHOLD ANALYSIS")
print("=" * 70)


print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


required_columns = [
    "SampleID",
    "Time_h",
    "Replicate",
    "Treatment"
] + SENSORS


missing = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing:

    print("\nERROR: Missing columns:")
    print(missing)

    print("\nColumns actually found:")
    print(df.columns.tolist())

    raise SystemExit


print("\nAll required columns found.")


for sensor in SENSORS:

    df[sensor] = pd.to_numeric(
        df[sensor],
        errors="coerce"
    )


df["Time_h"] = pd.to_numeric(
    df["Time_h"],
    errors="coerce"
)


print("\nTreatment distribution:")

print(
    df["Treatment"].value_counts()
)


print("\nRemoving Mechanical treatment...")

df = df[
    df["Treatment"].isin(
        [
            "Control",
            "Low",
            "Medium",
            "High"
        ]
    )
].copy()


df["positive"] = (
    df["Treatment"].isin(
        [
            "Low",
            "Medium",
            "High"
        ]
    )
).astype(int)


print("\nClass distribution:")

print(
    df["positive"].value_counts()
)


print("\nAggregating sensor readings...")


sample_df = (
    df.groupby(
        [
            "SampleID",
            "Time_h",
            "Replicate",
            "Treatment",
            "positive"
        ],
        as_index=False
    )[SENSORS]
    .mean()
)


print("\nAggregated dataset shape:")

print(
    sample_df.shape
)


results = []


roc_data = {}


print("\n")
print("=" * 70)
print("CALCULATING SENSOR THRESHOLDS")
print("=" * 70)


for sensor in SENSORS:

    print(
        f"\nProcessing {sensor}..."
    )


    data = sample_df[
        [
            sensor,
            "positive"
        ]
    ].dropna()


    y = data[
        "positive"
    ].values


    x = data[
        sensor
    ].values


    if len(np.unique(y)) < 2:

        print(
            "Not enough classes."
        )

        continue


    fpr, tpr, thresholds = roc_curve(
        y,
        x
    )


    roc_auc = auc(
        fpr,
        tpr
    )


    youden_j = (
        tpr - fpr
    )


    best_idx = np.argmax(
        youden_j
    )


    best_threshold = thresholds[
        best_idx
    ]


    sensitivity = tpr[
        best_idx
    ]


    specificity = 1 - fpr[
        best_idx
    ]


    results.append(
        {
            "Sensor": sensor,
            "Threshold": best_threshold,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Youden_J": youden_j[best_idx],
            "AUC": roc_auc
        }
    )


    roc_data[sensor] = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc
    }


    print(
        f"Threshold  : {best_threshold:.6f}"
    )

    print(
        f"Sensitivity : {sensitivity:.4f}"
    )

    print(
        f"Specificity : {specificity:.4f}"
    )

    print(
        f"Youden J    : {youden_j[best_idx]:.4f}"
    )

    print(
        f"AUC         : {roc_auc:.4f}"
    )


threshold_df = pd.DataFrame(
    results
)


threshold_df = threshold_df.sort_values(
    "AUC",
    ascending=False
).reset_index(
    drop=True
)


print("\n")
print("=" * 70)
print("FINAL SENSOR THRESHOLDS")
print("=" * 70)


print(
    threshold_df.to_string(
        index=False
    )
)


threshold_df.to_csv(
    "sensor_thresholds.csv",
    index=False
)


print(
    "\nSaved: sensor_thresholds.csv"
)


best_sensor = threshold_df.iloc[0]


print("\n")
print("=" * 70)
print("BEST SENSOR")
print("=" * 70)


print(
    f"Sensor      : {best_sensor['Sensor']}"
)

print(
    f"Threshold   : {best_sensor['Threshold']:.6f}"
)

print(
    f"Sensitivity : {best_sensor['Sensitivity']:.4f}"
)

print(
    f"Specificity : {best_sensor['Specificity']:.4f}"
)

print(
    f"AUC         : {best_sensor['AUC']:.4f}"
)


print("\nGenerating ROC graphs...")


plt.figure(
    figsize=(12, 8)
)


for sensor in roc_data:

    plt.plot(
        roc_data[sensor]["fpr"],
        roc_data[sensor]["tpr"],
        linewidth=2,
        label=(
            f"{sensor} "
            f"(AUC={roc_data[sensor]['auc']:.3f})"
        )
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1
)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curves - Olfact-AI Sensors"
)

plt.grid(
    alpha=0.3
)

plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.tight_layout()


plt.savefig(
    "sensor_roc_curves.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


print(
    "Saved: sensor_roc_curves.png"
)


print(
    "\nGenerating threshold graph..."
)


plt.figure(
    figsize=(12, 6)
)


plt.bar(
    threshold_df["Sensor"],
    threshold_df["Threshold"]
)


plt.xlabel(
    "Sensor"
)

plt.ylabel(
    "Optimal Sensor Threshold"
)

plt.title(
    "Data-Driven Sensor Thresholds"
)

plt.xticks(
    rotation=45
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    "sensor_thresholds.png",
    dpi=300
)


plt.show()


print(
    "Saved: sensor_thresholds.png"
)


print("\n")
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(
    "\nFiles generated:"
)

print(
    "1. sensor_thresholds.csv"
)

print(
    "2. sensor_roc_curves.png"
)

print(
    "3. sensor_thresholds.png"
)