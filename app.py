
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("machine_failure_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="🏭",
    layout="wide"
)

st.title("AI-Based Predictive Maintenance")
st.write("Monitor machine parameters and predict potential failures.")

# Model performance from your latest test results
st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("Test Accuracy", "98.05%")
col2.metric("Failure Precision", "91.43%")
col3.metric("Failure Recall", "47.06%")

st.caption(
    "Metrics are from your previous test evaluation. "
    "They are not a guarantee of future performance."
)

with st.expander("View test confusion matrix"):
    st.write("Results from 2,000 test records")

    confusion = pd.DataFrame(
        [[1929, 3], [36, 32]],
        index=["Actual Normal", "Actual Failure"],
        columns=["Predicted Normal", "Predicted Failure"]
    )

    st.dataframe(confusion)

    st.bar_chart(
        pd.DataFrame({
            "Number of records": [1929, 3, 36, 32]
        }, index=[
            "Correct normal",
            "False alarm",
            "Missed failure",
            "Detected failure"
        ])
    )

st.divider()
st.subheader("Machine Parameters")

machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
air_temp = st.number_input("Air Temperature (K)", value=300.0)
process_temp = st.number_input("Process Temperature (K)", value=310.0)
rpm = st.number_input("Rotational Speed (rpm)", value=1500)
torque = st.number_input("Torque (Nm)", value=40.0)
tool_wear = st.number_input("Tool Wear (minutes)", value=100)

if st.button("Predict Machine Failure", type="primary"):

    row = {
        "Type_L": int(machine_type == "L"),
        "Type_M": int(machine_type == "M"),
        "Type_H": int(machine_type == "H"),
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }

    input_data = pd.DataFrame([row])
    input_data = input_data[model_columns]

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Potential Machine Failure Detected")
        st.write("Please arrange a machine inspection.")
    else:
        st.success("No Machine Failure Predicted")
        st.write(
            "The model does not predict a failure for these inputs."
        )

    st.subheader("Entered Machine Readings")

    readings = pd.DataFrame({
        "Parameter": [
            "Air Temperature (K)",
            "Process Temperature (K)",
            "Rotational Speed (rpm)",
            "Torque (Nm)",
            "Tool Wear (minutes)"
        ],
        "Value": [
            air_temp,
            process_temp,
            rpm,
            torque,
            tool_wear
        ]
    })

    st.dataframe(readings, hide_index=True)

    st.warning(
        "This is an AI prediction, not a guarantee of machine safety."
    )
