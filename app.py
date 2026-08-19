import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import gdown
import os


st.set_page_config(
    page_title="Olfact-AI",
    page_icon="🌱",
    layout="wide"
)

MODEL_PATH = "olfact_model.pkl"
FEATURE_PATH = "feature_columns.pkl"
CONFIG_PATH = "model_config.pkl"
DATA_PATH = "data.csv"



DATA_FILE_ID = "1MH9Qu8hO1uS3eGI8jUFTTQynMYU0NAI9"


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


@st.cache_resource
def load_model_files():

    model = joblib.load(
        MODEL_PATH
    )

    feature_columns = joblib.load(
        FEATURE_PATH
    )

    config = joblib.load(
        CONFIG_PATH
    )

    return (
        model,
        feature_columns,
        config
    )


model, FEATURE_COLUMNS, CONFIG = load_model_files()

THRESHOLD = CONFIG[
    "probability_threshold"
]


@st.cache_data
def load_data():

    if not os.path.exists(DATA_PATH):

        url = (
            "https://drive.google.com/uc?id="
            + DATA_FILE_ID
        )

        with st.spinner(
            "Downloading sensor dataset..."
        ):

            gdown.download(
                url,
                DATA_PATH,
                quiet=False
            )

    df = pd.read_csv(
        DATA_PATH
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    return df


st.title(
    "🌱 Olfact-AI"
)

st.subheader(
    "Early Pest Infestation Detection"
)

st.write(
    "Sensor-based early detection "
    "of pest infestation using machine learning."
)


st.sidebar.header(
    "Detection Settings"
)


threshold = st.sidebar.slider(
    "Detection Threshold",
    min_value=0.05,
    max_value=0.95,
    value=float(THRESHOLD),
    step=0.05
)


st.sidebar.write(
    f"Selected threshold: "
    f"{threshold * 100:.0f}%"
)


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


st.header(
    "Experiment Information"
)


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


if st.button(
    "🔍 Predict Infection",
    type="primary",
    use_container_width=True
):

    feature_data = {}


    for feature in FEATURE_COLUMNS:

        feature_data[feature] = 0.0

        for sensor in SENSORS:

            if feature.startswith(
                sensor + "_"
            ):

                value = values[sensor]

                if feature.endswith(
                    "_mean"
                ):

                    feature_data[
                        feature
                    ] = value

                elif feature.endswith(
                    "_std"
                ):

                    feature_data[
                        feature
                    ] = 0.0

                elif feature.endswith(
                    "_min"
                ):

                    feature_data[
                        feature
                    ] = value

                elif feature.endswith(
                    "_max"
                ):

                    feature_data[
                        feature
                    ] = value

                break


    X = pd.DataFrame(
        [feature_data]
    )


    X = X[
        FEATURE_COLUMNS
    ]


    X = X.fillna(0)


    probability = model.predict_proba(
        X
    )[0][1]


    infected = (
        probability >= threshold
    )


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
            "Detection Threshold",
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


    st.header(
        "📈 Current Sensor Response"
    )


    sensor_df = pd.DataFrame(
        {
            "Sensor": SENSORS,
            "Response": [
                values[sensor]
                for sensor in SENSORS
            ]
        }
    )


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


    st.pyplot(
        fig
    )


    st.divider()


    st.header(
        "📊 Current Infection Probability"
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
        label=(
            f"Threshold = "
            f"{threshold * 100:.0f}%"
        )
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


    st.pyplot(
        fig2
    )


    st.divider()


    