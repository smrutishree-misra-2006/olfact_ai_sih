import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Olfact-AI",
    page_icon="🌱",
    layout="wide"
)


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

SENSOR_GASES = {
    "TGS2600": "H₂(General Air Quality)",
    "TGS2602": "VOC, Odorant Gases (Indoor Air Quality)",
    "TGS822": "Ethanol",
    "MQ3": "Alcohol Vapours",
    "MQ135": "NOₓ, Smoke",
    "MQ138": "Toluene, Acetone",
    "MiCS_NO2": "NO₂",
    "MiCS_NH3": "NH₃",
    "MiCS_CO": "CO"
}

SENSORS = list(
    SENSOR_THRESHOLDS.keys()
)


st.title("🌱 Olfact-AI")

st.subheader(
    "Early Pest Infestation Detection"
)

st.write(
    "Enter the sensor readings to determine "
    "whether the sensor response exceeds the "
    "data-driven infestation thresholds."
)


st.sidebar.header(
    "Experiment Information"
)








st.subheader(
    "🧪 Enter Sensor Readings"
)


cols = st.columns(3)


sensor_values = {}


for i, sensor in enumerate(SENSORS):

    with cols[i % 3]:

        st.markdown(
            f"**{sensor}**  \n"
            f"🧪 *Detects: {SENSOR_GASES[sensor]}*"
        )

        sensor_values[sensor] = st.number_input(
            f"{sensor} reading",
            min_value=0.0,
            value=float(SENSOR_THRESHOLDS[sensor] * 0.9),
            format="%.6f",
            key=sensor
        )

if st.button(
    "🔍 Analyze Infestation",
    type="primary"
):

    results = []


    for sensor in SENSORS:

        value = sensor_values[sensor]

        threshold = SENSOR_THRESHOLDS[sensor]

        exceeded = value >= threshold

        difference = value - threshold

        percentage = (
            difference / threshold
        ) * 100


        results.append(
            {
                "Sensor": sensor,
                "Reading": value,
                "Threshold": threshold,
                "Exceeded": exceeded,
                "Difference": difference,
                "Excess (%)": percentage
            }
        )


    result_df = pd.DataFrame(
        results
    )


    exceeded_count = int(
        result_df["Exceeded"].sum()
    )


    total_sensors = len(SENSORS)


    if exceeded_count <= 2:

        status = "NO INFESTATION"
        status_message = (
            "Sensor responses are mostly below "
            "the data-driven thresholds."
        )


    elif exceeded_count <= 5:

        status = "WARNING"
        status_message = (
            "Several sensor responses have exceeded "
            "their data-driven thresholds."
        )


    else:

        status = "INFESTATION DETECTED"
        status_message = (
            "Most sensor responses have exceeded "
            "their data-driven thresholds."
        )


    st.divider()


    st.subheader(
        "📊 Prediction"
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "Thresholds Exceeded",
            f"{exceeded_count}/{total_sensors}"
        )


    with c2:

        st.metric(
            "Time",
            f"{time_h:g} hours"
        )


    with c3:

        st.metric(
            "Sensors",
            f"{total_sensors}"
        )


    if status == "INFESTATION DETECTED":

        st.error(
            f"🚨 {status}"
        )

    elif status == "WARNING":

        st.warning(
            f"⚠️ {status}"
        )

    else:

        st.success(
            f"✅ {status}"
        )


    st.info(
        status_message
    )


    st.subheader(
        "📈 Sensor Response vs Detection Threshold"
    )


    plot_df = result_df[
        [
            "Sensor",
            "Reading",
            "Threshold"
        ]
    ].copy()


    plot_df = plot_df.set_index(
        "Sensor"
    )


    fig, ax = plt.subplots(
        figsize=(13, 6)
    )


    x = range(
        len(plot_df)
    )


    ax.bar(
        [i - 0.2 for i in x],
        plot_df["Reading"],
        width=0.4,
        label="Current Sensor Reading"
    )


    ax.bar(
        [i + 0.2 for i in x],
        plot_df["Threshold"],
        width=0.4,
        label="Detection Threshold"
    )


    ax.set_xticks(
        list(x)
    )


    ax.set_xticklabels(
        plot_df.index,
        rotation=45
    )


    ax.set_ylabel(
        "Sensor Response"
    )


    ax.set_title(
        "Current Sensor Response vs Data-Driven Threshold"
    )


    ax.legend()


    ax.grid(
        axis="y",
        alpha=0.3
    )


    plt.tight_layout()


    st.pyplot(
        fig
    )


    st.subheader(
        "🔬 Sensor Analysis"
    )


    display_df = result_df.copy()


    display_df["Reading"] = display_df[
        "Reading"
    ].round(6)


    display_df["Threshold"] = display_df[
        "Threshold"
    ].round(6)


    display_df["Difference"] = display_df[
        "Difference"
    ].round(6)


    display_df["Excess (%)"] = display_df[
        "Excess (%)"
    ].round(2)


    display_df = display_df.rename(
        columns={
            "Sensor": "Sensor",
            "Reading": "Current Reading",
            "Threshold": "Detection Threshold",
            "Exceeded": "Threshold Exceeded",
            "Difference": "Amount Above/Below",
            "Excess (%)": "Difference (%)"
        }
    )


    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


    st.subheader(
        "🚨 Exceeded Sensors"
    )


    exceeded_df = result_df[
        result_df["Exceeded"]
    ].copy()


    if len(exceeded_df) == 0:

        st.success(
            "No sensor exceeded its "
            "data-driven detection threshold."
        )

    else:

        for _, row in exceeded_df.iterrows():

            st.write(
                f"**{row['Sensor']}** — "
                f"Reading: `{row['Reading']:.6f}` | "
                f"Threshold: `{row['Threshold']:.6f}` | "
                f"Exceeded by: "
                f"`{row['Difference']:.6f}` "
                f"({row['Excess (%)']:.2f}%)"
            )


    st.subheader(
        "📊 Threshold Exceedance"
    )


    exceed_plot = result_df.copy()


    exceed_plot["Excess"] = (
        exceed_plot["Reading"]
        - exceed_plot["Threshold"]
    )


    fig2, ax2 = plt.subplots(
        figsize=(13, 6)
    )


    ax2.bar(
        exceed_plot["Sensor"],
        exceed_plot["Excess"]
    )


    ax2.axhline(
        0,
        linewidth=1
    )


    ax2.set_xlabel(
        "Sensor"
    )


    ax2.set_ylabel(
        "Reading - Threshold"
    )


    ax2.set_title(
        "Sensor Threshold Exceedance"
    )


    ax2.tick_params(
        axis="x",
        rotation=45
    )


    ax2.grid(
        axis="y",
        alpha=0.3
    )


    plt.tight_layout()


    st.pyplot(
        fig2
    )


    st.subheader(
        "⏱️ Detection Summary"
    )


    st.write(
        f"At **{time_h:g} hours**, "
        f"**{exceeded_count} out of {total_sensors} sensors** "
        f"exceeded their data-driven detection thresholds."
    )


    if status == "INFESTATION DETECTED":

        st.error(
            "🚨 INFESTATION DETECTED"
        )

    elif status == "WARNING":

        st.warning(
            "⚠️ WARNING — possible infestation"
        )

    else:

        st.success(
            "✅ NO INFESTATION"
        )


    csv = result_df.to_csv(
        index=False
    )


    st.download_button(
        "Download Sensor Analysis",
        data=csv,
        file_name="olfact_ai_sensor_analysis.csv",
        mime="text/csv"
    )


st.divider()


