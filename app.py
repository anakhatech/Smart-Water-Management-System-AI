import streamlit as st
import pandas as pd
import joblib


# =============================
# PAGE CONFIGURATION
# =============================

st.set_page_config(
    page_title="Smart Water Management AI",
    page_icon="💧",
    layout="wide"
)


# =============================
# LOAD AI MODEL
# =============================

@st.cache_resource
def load_model():
    model = joblib.load("water_leakage_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")

    return model, feature_columns


model, feature_columns = load_model()


# =============================
# CUSTOM CSS
# =============================

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(135deg, #0b1f2a, #12394a);
    }

    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #7dd3fc;
    }

    .subtitle {
        font-size: 18px;
        color: #cbd5e1;
        margin-bottom: 30px;
    }

    .result-card {
        padding: 30px;
        border-radius: 20px;
        background: #17384a;
        border: 1px solid #2c5364;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =============================
# SIDEBAR
# =============================

st.sidebar.title("💧 Water AI")

st.sidebar.write("Smart Water Management System")

st.sidebar.divider()


page = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "🔍 Leakage Prediction"]
)


st.sidebar.divider()

st.sidebar.success("🟢 AI System Online")

st.sidebar.caption(
    "AI-powered water pipeline monitoring system."
)


# =============================
# DASHBOARD PAGE
# =============================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">💧 Smart Water Management System</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">AI-powered pipeline monitoring and water leakage prediction</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("📊 System Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🤖 AI Model",
            "Active"
        )

    with col2:
        st.metric(
            "📡 System Status",
            "Monitoring"
        )

    with col3:
        st.metric(
            "💧 Leakage Detection",
            "Enabled"
        )

    st.divider()

    st.subheader("🌊 About the System")

    st.write("""
    The Smart Water Management System uses Artificial Intelligence
    to analyze pipeline sensor data and predict potential water leakage.

    The system evaluates:

    - Pressure
    - Flow Rate
    - Temperature
    - Vibration
    - RPM
    - Operational Hours
    - Pipeline Zone, Block and Pipe information
    """)


# =============================
# LEAKAGE PREDICTION PAGE
# =============================

elif page == "🔍 Leakage Prediction":

    st.markdown(
        '<div class="main-title">🔍 Water Leakage Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Enter pipeline sensor data to analyze potential water leakage.</div>',
        unsafe_allow_html=True
    )

    st.divider()


    # =============================
    # SENSOR DATA
    # =============================

    st.subheader("📡 Sensor Data")

    col1, col2, col3 = st.columns(3)


    with col1:

        pressure = st.number_input(
            "Pressure",
            value=65.0
        )

        flow_rate = st.number_input(
            "Flow Rate",
            value=70.0
        )


    with col2:

        temperature = st.number_input(
            "Temperature",
            value=100.0
        )

        vibration = st.number_input(
            "Vibration",
            value=3.0
        )


    with col3:

        rpm = st.number_input(
            "RPM",
            value=2000.0
        )

        operational_hours = st.number_input(
            "Operational Hours",
            value=3000
        )


    # =============================
    # PIPELINE DETAILS
    # =============================

    st.divider()

    st.subheader("🏗️ Pipeline Details")

    col4, col5, col6 = st.columns(3)


    with col4:

        zone = st.selectbox(
            "Zone",
            [
                "Zone_1",
                "Zone_2",
                "Zone_3",
                "Zone_4",
                "Zone_5"
            ]
        )


    with col5:

        block = st.selectbox(
            "Block",
            [
                "Block_1",
                "Block_2",
                "Block_3",
                "Block_4",
                "Block_5"
            ]
        )


    with col6:

        pipe = st.selectbox(
            "Pipe",
            [
                "Pipe_1",
                "Pipe_2",
                "Pipe_3",
                "Pipe_4",
                "Pipe_5"
            ]
        )


    selected_pipeline = f"{zone} → {block} → {pipe}"


    st.info(
        f"📍 Selected Pipeline: **{selected_pipeline}**"
    )


    st.divider()


    # =============================
    # PREDICTION BUTTON
    # =============================

    predict_button = st.button(
        "🚀 ANALYZE WATER LEAKAGE",
        use_container_width=True
    )


    # =============================
    # AI PREDICTION
    # =============================

    if predict_button:

        with st.spinner(
            "🤖 AI is analyzing the pipeline data..."
        ):

            # Create input dataframe using
            # the exact columns used during training

            input_data = pd.DataFrame(
                0,
                index=[0],
                columns=feature_columns
            )


            # =============================
            # ADD SENSOR VALUES
            # =============================

            numeric_values = {
                "Pressure": pressure,
                "Flow_Rate": flow_rate,
                "Temperature": temperature,
                "Vibration": vibration,
                "RPM": rpm,
                "Operational_Hours": operational_hours
            }


            for column, value in numeric_values.items():

                if column in input_data.columns:

                    input_data[column] = value


            # =============================
            # ZONE ENCODING
            # =============================

            zone_column = f"Zone_{zone}"

            if zone_column in input_data.columns:

                input_data[zone_column] = 1


            # =============================
            # BLOCK ENCODING
            # =============================

            block_column = f"Block_{block}"

            if block_column in input_data.columns:

                input_data[block_column] = 1


            # =============================
            # PIPE ENCODING
            # =============================

            pipe_column = f"Pipe_{pipe}"

            if pipe_column in input_data.columns:

                input_data[pipe_column] = 1


            # =============================
            # LOCATION CODE ENCODING
            # =============================

            location_column = (
                f"Location_Code_"
                f"{zone}_{block}_{pipe}"
            )


            if location_column in input_data.columns:

                input_data[location_column] = 1


            # =============================
            # PREDICTION
            # =============================

            prediction = model.predict(
                input_data
            )[0]


            probability = model.predict_proba(
                input_data
            )[0][1]


        # =============================
        # RESULT
        # =============================

        st.divider()

        st.subheader("🤖 AI Prediction Result")


        result_col1, result_col2 = st.columns([2, 1])


        with result_col1:

            if prediction == 1:

                st.error(
                    "🚨 WATER LEAKAGE DETECTED!"
                )


                st.markdown(
                    f"""
                    <div class="result-card">
                        <h2>⚠️ Pipeline Requires Immediate Attention</h2>

                        <p>
                            <b>Affected Pipeline:</b>
                            {selected_pipeline}
                        </p>

                        <p>
                            The AI model detected a potential water leakage
                            based on the current sensor readings.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            else:

                st.success(
                    "✅ NO WATER LEAKAGE DETECTED"
                )


                st.markdown(
                    f"""
                    <div class="result-card">
                        <h2>🟢 Pipeline Status Normal</h2>

                        <p>
                            <b>Analyzed Pipeline:</b>
                            {selected_pipeline}
                        </p>

                        <p>
                            The AI model indicates that the current sensor
                            readings do not show significant leakage risk.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


        with result_col2:

            st.metric(
                "💧 Leakage Probability",
                f"{probability * 100:.1f}%"
            )


            st.progress(
                min(
                    int(probability * 100),
                    100
                )
            )


            st.caption(
                f"Selected: {selected_pipeline}"
            ) 