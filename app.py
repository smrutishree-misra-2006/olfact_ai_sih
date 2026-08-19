import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path


st.set_page_config(
    page_title="Olfact-AI",
    page_icon="🌱",
    layout="wide"
)


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "olfact_model.pkl"
FEATURE_PATH = BASE_DIR / "feature_columns.pkl"
CONFIG_PATH = BASE_DIR / "model_config.pkl"
SENSORS_PATH = BASE_DIR / "sensors.pkl"


SENSOR_GASES = {
    "TGS2600": "H₂, CO, Alcohol",
    "TGS2602": "VOCs, NH₃, H₂S",
    "TGS822": "Ethanol, Methanol",
    "MQ3": "Alcohol / Ethanol",
    "MQ135": "NH₃, NOₓ, Alcohol, Smoke",
    "MQ138": "VOCs, Toluene, Acetone, Alcohol, H₂",
    "MiCS_NO2": "NO₂",
    "MiCS_NH3": "NH₃",
    "MiCS_CO": "CO"
}


SENSOR_THRESHOLDS = {
    "TGS2600": 0.986141,
    "TGS2602": 1.069869,
    "TGS822": 0.908817,
    "MQ3": 0.863072,
    "MQ135": 0.940894,
    "MQ138": 0.975829,
    "MiCS_NO2": 0.036819,
    "MiCS_NH3": 0.053022,
    "MiCS_CO": 0.642735
}


SENSOR_NAMES = list(SENSOR_THRESHOLDS.keys())


@st.cache_resource
def load_model_files():

    model = joblib.load(MODEL_PATH)

    feature_columns = joblib.load(FEATURE_PATH)

    config = joblib.load(CONFIG_PATH)

    try:
        sensors = joblib.load(SENSORS_PATH)
    except:
        sensors = SENSOR_NAMES

    return model, feature_columns, config, sensors


for file_path in [
    MODEL_PATH,
    FEATURE_PATH,
    CONFIG_PATH
]:

    if not file_path.exists():

        st.error(
            f"Required file missing: {file_path.name}"
        )

        st.stop()


try:

    model, FEATURE_COLUMNS, CONFIG, SENSORS = load_model_files()

except Exception as e:

    st.error("Could not load the trained model.")

    st.exception(e)

    st.stop()


st.title("🌱 Olfact-AI")

st.subheader(
    "Early Pest Infestation Detection"
)

st.write(
    "Enter the sensor readings collected from the device. "
    "Olfact-AI compares the gas response with learned sensor "
    "thresholds and uses machine learning to estimate infestation."
)


st.divider()


st.subheader("🧪 Sensor Readings")


st.info(
    "Enter the readings produced by your sensor device. "
    "The threshold values below were obtained from the experimental dataset "
    "using ROC/Youden analysis."
)


sensor_values = {}


cols = st.columns(3)


for i, sensor in enumerate(SENSOR_NAMES):

    with cols[i % 3]:

        st.markdown(
            f"### {sensor}"
        )

        st.caption(
            f"🧪 Detects: {SENSOR_GASES[sensor]}"
        )

        sensor_values[sensor] = st.number_input(
            f"{sensor} reading",
            min_value=0.0,
            value=0.0,
            step=0.001,
            format="%.6f",
            key=f"input_{sensor}"
        )

        st.caption(
            f"Learned threshold: "
            f"{SENSOR_THRESHOLDS[sensor]:.6f}"
        )


st.divider()


st.subheader("🕒 Experiment Information")


col1, col2 = st.columns(2)


with col1:

    time_h = st.number_input(
        "Time after infection (hours)",
        min_value=0.0,
        value=1.0,
        step=1.0
    )


with col2:

    treatment = st.selectbox(
        "Treatment",
        [
            "Control",
            "Low",
            "Medium",
            "High",
            "Mechanical"
        ]
    )


st.divider()


if st.button(
    "🔍 Analyze Plant",
    type="primary",
    use_container_width=True
):

    threshold_results = []

    for sensor in SENSOR_NAMES:

        value = sensor_values[sensor]

        threshold = SENSOR_THRESHOLDS[sensor]

        exceeded = value >= threshold

        threshold_results.append({
            "Sensor": sensor,
            "Gas detected": SENSOR_GASES[sensor],
            "Reading": value,
            "Threshold": threshold,
            "Exceeded": exceeded
        })


    threshold_df = pd.DataFrame(
        threshold_results
    )


    exceeded_count = int(
        threshold_df["Exceeded"].sum()
    )


    total_sensors = len(SENSOR_NAMES)


    threshold_ratio = (
        exceeded_count / total_sensors
    )


    try:

        feature_data = {}

        for feature in FEATURE_COLUMNS:

            if feature == "Time_h":

                feature_data[feature] = time_h

            elif feature.endswith("_mean"):

                sensor = feature.replace(
                    "_mean",
                    ""
                )

                if sensor in sensor_values:

                    feature_data[feature] = (
                        sensor_values[sensor]
                    )

                else:

                    feature_data[feature] = 0.0

            elif feature.endswith("_std"):

                sensor = feature.replace(
                    "_std",
                    ""
                )

                if sensor in sensor_values:

                    feature_data[feature] = 0.0

                else:

                    feature_data[feature] = 0.0

            elif feature.endswith("_min"):

                sensor = feature.replace(
                    "_min",
                    ""
                )

                if sensor in sensor_values:

                    feature_data[feature] = (
                        sensor_values[sensor]
                    )

                else:

                    feature_data[feature] = 0.0

            elif feature.endswith("_max"):

                sensor = feature.replace(
                    "_max",
                    ""
                )

                if sensor in sensor_values:

                    feature_data[feature] = (
                        sensor_values[sensor]
                    )

                else:

                    feature_data[feature] = 0.0

            else:

                feature_data[feature] = 0.0


        X = pd.DataFrame(
            [feature_data],
            columns=FEATURE_COLUMNS
        )


        X = X.replace(
            [np.inf, -np.inf],
            np.nan
        ).fillna(0)


        probability = float(
            model.predict_proba(X)[0][1]
        )


    except Exception as e:

        st.error(
            "Model prediction could not be generated."
        )

        st.exception(e)

        st.stop()


    try:

        model_threshold = float(
            CONFIG.get(
                "probability_threshold",
                0.60
            )
        )

    except:

        model_threshold = 0.60


    sensor_condition = (
        threshold_ratio >= 0.50
    )


    model_condition = (
        probability >= model_threshold
    )


    infestation_detected = (
        sensor_condition and model_condition
    )


    st.divider()

    st.subheader("📊 Prediction")


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "Infection Probability",
            f"{probability * 100:.2f}%"
        )


    with c2:

        st.metric(
            "ML Threshold",
            f"{model_threshold * 100:.0f}%"
        )


    with c3:

        st.metric(
            "Sensors Above Threshold",
            f"{exceeded_count}/{total_sensors}"
        )


    with c4:

        st.metric(
            "Threshold Ratio",
            f"{threshold_ratio * 100:.1f}%"
        )


    if infestation_detected:

        st.error(
            "🚨 INFESTATION DETECTED"
        )

        st.write(
            f"{exceeded_count} out of "
            f"{total_sensors} sensors exceeded "
            "their learned sensor threshold, and "
            "the ML model probability is above its "
            "detection threshold."
        )

    else:

        st.success(
            "✅ NO INFESTATION"
        )

        st.write(
            f"{exceeded_count} out of "
            f"{total_sensors} sensors exceeded "
            "their learned sensor threshold, and "
            "the combined evidence did not satisfy "
            "the infestation detection criteria."
        )


    st.divider()


    st.subheader(
        "📈 Sensor Response vs Learned Threshold"
    )


    plot_df = threshold_df.copy()

    plot_df["Reading"] = pd.to_numeric(
        plot_df["Reading"]
    )

    plot_df["Threshold"] = pd.to_numeric(
        plot_df["Threshold"]
    )


    fig, ax = plt.subplots(
        figsize=(13, 6)
    )


    x = np.arange(
        len(plot_df)
    )

    width = 0.35


    ax.bar(
        x - width / 2,
        plot_df["Reading"],
        width,
        label="Current Reading"
    )


    ax.bar(
        x + width / 2,
        plot_df["Threshold"],
        width,
        label="Learned Threshold"
    )


    ax.set_xticks(x)

    ax.set_xticklabels(
        plot_df["Sensor"],
        rotation=30,
        ha="right"
    )


    ax.set_ylabel(
        "Sensor Response"
    )

    ax.set_xlabel(
        "Sensor"
    )


    ax.set_title(
        "Current Sensor Response vs Learned Threshold"
    )


    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3
    )


    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )


    st.divider()


    st.subheader(
        "🧪 Gas Detection Status"
    )


    status_df = threshold_df[
        [
            "Sensor",
            "Gas detected",
            "Reading",
            "Threshold",
            "Exceeded"
        ]
    ].copy()


    status_df["Status"] = status_df[
        "Exceeded"
    ].apply(
        lambda x:
        "⚠ Above threshold"
        if x
        else "✓ Normal"
    )


    status_df = status_df.drop(
        columns=["Exceeded"]
    )


    status_df = status_df.rename(
        columns={
            "Sensor": "Sensor",
            "Gas detected": "Gas detected",
            "Reading": "Current reading",
            "Threshold": "Learned threshold",
            "Status": "Status"
        }
    )


    st.dataframe(
        status_df,
        use_container_width=True,
        hide_index=True
    )


    st.divider()


    st.subheader(
        "📊 Infection Probability"
    )


    fig2, ax2 = plt.subplots(
        figsize=(10, 5)
    )


    ax2.bar(
        ["Infection Probability"],
        [probability]
    )


    ax2.axhline(
        model_threshold,
        linestyle="--",
        linewidth=2,
        label=(
            f"ML Threshold "
            f"({model_threshold * 100:.0f}%)"
        )
    )


    ax2.set_ylim(
        0,
        1
    )


    ax2.set_ylabel(
        "Probability"
    )


    ax2.set_title(
        "Olfact-AI Infection Probability"
    )


    ax2.legend()

    ax2.grid(
        axis="y",
        alpha=0.3
    )


    plt.tight_layout()

    st.pyplot(
        fig2,
        use_container_width=True
    )


    st.divider()


    st.subheader(
        "⏱️ Detection Time"
    )


    if infestation_detected:

        st.success(
            f"Estimated detection point: "
            f"{time_h:g} hours after infection"
        )

    else:

        st.info(
            "No infestation detected at the "
            f"provided {time_h:g}-hour measurement."
        )


    st.divider()


    result_df = threshold_df.copy()

    result_df["Infection Probability"] = (
        probability * 100
    )

    result_df["ML Threshold"] = (
        model_threshold * 100
    )

    result_df["Time_h"] = time_h

    result_df["Treatment"] = treatment

    result_df["Infestation"] = (
        "INFESTATION DETECTED"
        if infestation_detected
        else "NO INFESTATION"
    )


    csv = result_df.to_csv(
        index=False
    )


    st.download_button(
        "⬇️ Download Analysis",
        data=csv,
        file_name="olfact_ai_analysis.csv",
        mime="text/csv",
        use_container_width=True
    )


st.divider()


st.caption(
    "Olfact-AI uses multi-sensor gas-response patterns "
    "and machine learning for experimental pest-infestation "
    "detection."
)
