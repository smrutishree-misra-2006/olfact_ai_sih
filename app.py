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


MODEL_PATH = r"C:\final\olfact_model.pkl"
FEATURE_PATH = r"C:\final\feature_columns.pkl"
CONFIG_PATH = r"C:\final\model_config.pkl"

DATA_FILE_ID = "1MH9Qu8hO1uS3eGI8jUFTTQynMYU0NAI9"
DATA_PATH = r"C:\final\data.csv"


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


    st.header(
        "⏱️ Detection Time Analysis"
    )


    try:

        df = load_data()


        required_columns = [
            "Time_h",
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


        missing_columns = [
            col
            for col in required_columns
            if col not in df.columns
        ]


        if missing_columns:

            st.error(
                "Dataset is missing these columns:"
            )

            st.write(
                missing_columns
            )

            st.write(
                "Columns found in dataset:"
            )

            st.write(
                df.columns.tolist()
            )

            st.stop()


        for sensor in SENSORS:

            df[sensor] = pd.to_numeric(
                df[sensor],
                errors="coerce"
            )


        df["Time_h"] = pd.to_numeric(
            df["Time_h"],
            errors="coerce"
        )


        df = df.dropna(
            subset=SENSORS + ["Time_h"]
        )


        rows = []


        for t, group in df.groupby(
            "Time_h"
        ):

            row = {
                "Time_h": t
            }


            for sensor in SENSORS:

                v = group[
                    sensor
                ].values


                row[
                    f"{sensor}_mean"
                ] = np.mean(v)


                row[
                    f"{sensor}_std"
                ] = np.std(v)


                row[
                    f"{sensor}_min"
                ] = np.min(v)


                row[
                    f"{sensor}_max"
                ] = np.max(v)


            rows.append(
                row
            )


        time_features = pd.DataFrame(
            rows
        )


        time_features = (
            time_features
            .sort_values("Time_h")
            .reset_index(drop=True)
        )


        X_time = (
            time_features[
                FEATURE_COLUMNS
            ]
            .fillna(0)
        )


        probabilities = (
            model.predict_proba(
                X_time
            )[:, 1]
        )


        time_features[
            "Probability"
        ] = probabilities


        detected = time_features[
            time_features[
                "Probability"
            ] >= threshold
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
                "No infestation threshold "
                "crossing was found."
            )


        st.subheader(
            "📈 Infection Probability vs Time"
        )


        fig3, ax3 = plt.subplots(
            figsize=(12, 5)
        )


        ax3.plot(
            time_features["Time_h"],
            time_features["Probability"],
            marker="o",
            linewidth=2,
            label="Infection Probability"
        )


        ax3.axhline(
            threshold,
            linestyle="--",
            linewidth=2,
            label=(
                f"Detection Threshold "
                f"({threshold * 100:.0f}%)"
            )
        )


        if len(detected) > 0:

            ax3.axvline(
                first_time,
                linestyle=":",
                linewidth=2,
                label=(
                    f"First Detection "
                    f"({first_time:g} h)"
                )
            )


        ax3.set_xlabel(
            "Time after infestation (hours)"
        )


        ax3.set_ylabel(
            "Infection Probability"
        )


        ax3.set_title(
            "Olfact-AI Infection Probability Over Time"
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


        st.pyplot(
            fig3
        )


        st.subheader(
            "Prediction Table"
        )


        display_df = time_features[
            [
                "Time_h",
                "Probability"
            ]
        ].copy()


        display_df[
            "Probability"
        ] = (
            display_df[
                "Probability"
            ] * 100
        )


        display_df[
            "Above Threshold"
        ] = (
            time_features[
                "Probability"
            ] >= threshold
        )


        display_df = (
            display_df
            .rename(
                columns={
                    "Time_h":
                        "Time (hours)",
                    "Probability":
                        "Infection Probability (%)"
                }
            )
        )


        st.dataframe(
            display_df,
            use_container_width=True
        )


        st.divider()


        st.header(
            "🧪 Sensor Response Over Time"
        )


        selected_sensor = st.selectbox(
            "Select Sensor",
            SENSORS
        )


        sensor_by_time = (
            df.groupby(
                "Time_h"
            )[selected_sensor]
            .mean()
            .reset_index()
        )


        fig4, ax4 = plt.subplots(
            figsize=(12, 5)
        )


        ax4.plot(
            sensor_by_time["Time_h"],
            sensor_by_time[
                selected_sensor
            ],
            marker="o",
            linewidth=2
        )


        ax4.set_xlabel(
            "Time after infestation (hours)"
        )


        ax4.set_ylabel(
            "Sensor Response"
        )


        ax4.set_title(
            f"{selected_sensor} Response vs Time"
        )


        ax4.grid(
            alpha=0.3
        )


        plt.tight_layout()


        st.pyplot(
            fig4
        )


    except Exception as e:

        st.error(
            "Could not generate detection-time analysis."
        )

        st.exception(e)


st.divider()


st.header(
    "ℹ️ Model Information"
)


st.write(
    f"Number of sensors: {len(SENSORS)}"
)


st.write(
    "Sensors:"
)


st.write(
    ", ".join(SENSORS)
)


st.write(
    f"Detection threshold: "
    f"{threshold * 100:.0f}%"
)