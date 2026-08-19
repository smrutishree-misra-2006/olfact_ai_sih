import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import gdown


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Olfact-AI",
    page_icon="🌱",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = "olfact_model.pkl"
FEATURE_PATH = "feature_columns.pkl"
SENSORS_PATH = "sensors.pkl"
CONFIG_PATH = "model_config.pkl"

# Google Drive file ID
DATA_FILE_ID = "1MH9Qu8hO1uS3eGI8jUFTTQynMYU0NAI9"

# Temporary location used by the deployed app
DATA_PATH = "data.csv"


# ============================================================
# LOAD MODEL FILES
# ============================================================

@st.cache_resource
def load_model_files():

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_PATH)
    sensors = joblib.load(SENSORS_PATH)
    config = joblib.load(CONFIG_PATH)

    return model, feature_columns, sensors, config


model, FEATURE_COLUMNS, SENSORS, CONFIG = load_model_files()

THRESHOLD = CONFIG["probability_threshold"]


# ============================================================
# DOWNLOAD DATASET FROM GOOGLE DRIVE
# ============================================================

@st.cache_data
def load_data():

    # Download only if the file doesn't already exist
    import os

    if not os.path.exists(DATA_PATH):

        url = f"https://drive.google.com/uc?id={DATA_FILE_ID}"

        with st.spinner(
            "Downloading sensor dataset... This may take a while on the first run."
        ):

            gdown.download(
                url,
                DATA_PATH,
                quiet=False
            )

    df = pd.read_csv(DATA_PATH)

    return df


# ============================================================
# TITLE
# ============================================================

st.title("🌱 Olfact-AI")

st.subheader(
    "Early Pest Infestation Detection"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Detection Settings"
)


threshold = st.sidebar.slider(
    "Detection Threshold",
    0.05,
    0.95,
    float(THRESHOLD),
    0.05
)


# ============================================================
# SENSOR INPUT
# ============================================================

st.header(
    "🧪 Enter Sensor Readings"
)


values = {}


cols = st.columns(3)


for i, sensor in enumerate(SENSORS):

    with cols[i % 3]:

        values[sensor] = st.number_input(
            sensor,
            value=0.0,
            format="%.6f"
        )


st.divider()


# ============================================================
# ADDITIONAL INPUTS
# ============================================================

col1, col2 = st.columns(2)


with col1:

    time_input = st.number_input(
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


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔍 Predict Infection",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # CREATE FEATURE VECTOR
    # --------------------------------------------------------

    feature_data = {}


    for feature in FEATURE_COLUMNS:

        found = False


        for sensor in SENSORS:

            if feature.startswith(sensor + "_"):

                value = values[sensor]


                if feature.endswith("_mean"):

                    feature_data[feature] = value


                elif feature.endswith("_std"):

                    feature_data[feature] = 0


                elif feature.endswith("_min"):

                    feature_data[feature] = value


                elif feature.endswith("_max"):

                    feature_data[feature] = value


                found = True

                break


        if not found:

            feature_data[feature] = 0


    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    X = pd.DataFrame(
        [feature_data]
    )[FEATURE_COLUMNS]


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    probability = model.predict_proba(X)[0][1]


    infected = probability >= threshold


    # ========================================================
    # PREDICTION RESULTS
    # ========================================================

    st.header(
        "📊 Prediction"
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "Infection Probability",
            f"{probability * 100:.2f}%"
        )


    with c2:

        st.metric(
            "Threshold",
            f"{threshold * 100:.0f}%"
        )


    with c3:

        if infected:

            st.error(
                "🚨 INFESTATION DETECTED"
            )

        else:

            st.success(
                "✅ NO INFESTATION DETECTED"
            )


    st.divider()


    # ========================================================
    # SENSOR RESPONSE GRAPH
    # ========================================================

    st.header(
        "📈 Sensor Response"
    )


    sensor_df = pd.DataFrame({

        "Sensor": SENSORS,

        "Response": [
            values[sensor]
            for sensor in SENSORS
        ]

    })


    fig, ax = plt.subplots(
        figsize=(12, 5)
    )


    ax.bar(
        sensor_df["Sensor"],
        sensor_df["Response"]
    )


    ax.set_xlabel(
        "Gas Sensors"
    )


    ax.set_ylabel(
        "Sensor Response"
    )


    ax.set_title(
        "Current Sensor Response"
    )


    ax.tick_params(
        axis="x",
        rotation=45
    )


    ax.grid(
        axis="y",
        alpha=0.3
    )


    plt.tight_layout()


    st.pyplot(fig)


    st.divider()


    # ========================================================
    # INFECTION PROBABILITY GRAPH
    # ========================================================

    st.header(
        "📈 Infection Probability"
    )


    fig2, ax2 = plt.subplots(
        figsize=(10, 5)
    )


    ax2.bar(
        ["Current Sample"],
        [probability]
    )


    ax2.axhline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold:.2f}"
    )


    ax2.set_ylim(
        0,
        1.05
    )


    ax2.set_ylabel(
        "Infection Probability"
    )


    ax2.set_title(
        "Olfact-AI Prediction"
    )


    ax2.legend()


    ax2.grid(
        axis="y",
        alpha=0.3
    )


    plt.tight_layout()


    st.pyplot(fig2)


    st.divider()


    # ========================================================
    # DETECTION TIME ANALYSIS
    # ========================================================

    st.header(
        "⏱️ Detection Time"
    )


    try:

        # Load dataset
        df = load_data()


        # ----------------------------------------------------
        # CLEAN SENSOR COLUMNS
        # ----------------------------------------------------

        for sensor in SENSORS:

            df[sensor] = pd.to_numeric(
                df[sensor],
                errors="coerce"
            )


        df["Time_h"] = pd.to_numeric(
            df["Time_h"],
            errors="coerce"
        )


        # ----------------------------------------------------
        # REMOVE INVALID ROWS
        # ----------------------------------------------------

        df = df.dropna(
            subset=SENSORS + ["Time_h"]
        )


        # ----------------------------------------------------
        # CALCULATE TIME FEATURES
        # ----------------------------------------------------

        rows = []


        for t, group in df.groupby("Time_h"):

            row = {}


            for sensor in SENSORS:

                v = group[sensor].values


                row[f"{sensor}_mean"] = np.mean(v)

                row[f"{sensor}_std"] = np.std(v)

                row[f"{sensor}_min"] = np.min(v)

                row[f"{sensor}_max"] = np.max(v)


            rows.append(row)


        time_features = pd.DataFrame(
            rows
        )


        # ----------------------------------------------------
        # CREATE MODEL INPUT
        # ----------------------------------------------------

        X_time = time_features[
            FEATURE_COLUMNS
        ].fillna(0)


        # ----------------------------------------------------
        # PREDICT PROBABILITIES
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            X_time
        )[:, 1]


        time_features["Probability"] = probabilities


        time_features["Time_h"] = sorted(
            df["Time_h"].unique()
        )


        # ----------------------------------------------------
        # FIND FIRST THRESHOLD CROSSING
        # ----------------------------------------------------

        detected = time_features[
            time_features["Probability"] >= threshold
        ]


        if len(detected) > 0:

            first_time = detected[
                "Time_h"
            ].iloc[0]


            st.success(
                f"🕒 First threshold crossing: "
                f"{first_time:g} hours"
            )


        else:

            st.warning(
                "No threshold crossing found."
            )


        # ====================================================
        # PROBABILITY VS TIME GRAPH
        # ====================================================

        fig3, ax3 = plt.subplots(
            figsize=(12, 5)
        )


        ax3.plot(
            time_features["Time_h"],
            time_features["Probability"],
            marker="o",
            linewidth=2
        )


        ax3.axhline(
            threshold,
            linestyle="--",
            linewidth=2,
            label="Detection Threshold"
        )


        ax3.set_xlabel(
            "Time (hours)"
        )


        ax3.set_ylabel(
            "Infection Probability"
        )


        ax3.set_title(
            "Infection Probability vs Time"
        )


        ax3.set_ylim(
            0,
            1.05
        )


        ax3.grid(
            alpha=0.3
        )


        ax3.legend()


        plt.tight_layout()


        st.pyplot(fig3)


    except Exception as e:

        st.error(
            "Could not generate time graph."
        )

        st.exception(e)