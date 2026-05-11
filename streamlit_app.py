import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(
    page_title="Healthcare Readmission Risk App",
    layout="wide"
)


@st.cache_resource
def load_model():
    return joblib.load("outputs/readmission_xgboost_model.joblib")


@st.cache_data
def load_data():
    return pd.read_csv("data/healthcare_readmission_dataset.csv")


model = load_model()
data = load_data()

st.title("Healthcare Readmission Risk Prediction")
st.write(
    "This app predicts 30-day hospital readmission risk using structured patient data "
    "and physician comments processed with TF-IDF."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Patient Risk Calculator",
    "Model Performance",
    "Risk Segmentation",
    "About the Model"
])


def classify_risk(probability):
    if probability < 0.30:
        return "Low Risk", "Routine follow-up recommended."
    elif probability < 0.60:
        return "Moderate Risk", "Schedule follow-up and monitor symptoms."
    else:
        return "High Risk", "Recommend early follow-up and care management review."


with tab1:
    st.header("Patient Risk Calculator")

    col1, col2 = st.columns(2)

    with col1:
        patient_age = st.number_input("Patient Age", 18, 90, 65)
        gender = st.selectbox("Gender", ["Female", "Male"])
        length_of_stay = st.number_input("Length of Stay", 1, 30, 4)
        prior_admissions = st.number_input("Prior Admissions", 0, 10, 1)
        emergency_visits = st.number_input("Emergency Visits", 0, 10, 1)
        medication_count = st.number_input("Medication Count", 0, 20, 5)

    with col2:
        diabetes = st.selectbox("Diabetes", [0, 1])
        hypertension = st.selectbox("Hypertension", [0, 1])
        heart_disease = st.selectbox("Heart Disease", [0, 1])
        smoker = st.selectbox("Smoker", [0, 1])
        bmi = st.number_input("BMI", 16.0, 55.0, 29.0)
        discharge_destination = st.selectbox(
            "Discharge Destination",
            ["Home", "Home with Support", "Rehab Facility", "Long-Term Care"]
        )

    physician_comments = st.text_area(
        "Physician Comments",
        "Patient continues smoking and reports shortness of breath."
    )

    if st.button("Predict Readmission Risk"):
        input_data = pd.DataFrame([{
            "patient_age": patient_age,
            "gender": gender,
            "length_of_stay": length_of_stay,
            "prior_admissions": prior_admissions,
            "emergency_visits": emergency_visits,
            "diabetes": diabetes,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "smoker": smoker,
            "bmi": bmi,
            "medication_count": medication_count,
            "discharge_destination": discharge_destination,
            "physician_comments": physician_comments
        }])

        probability = model.predict_proba(input_data)[0][1]
        risk_level, recommendation = classify_risk(probability)

        st.subheader("Prediction Results")
        st.metric("Readmission Probability", f"{probability:.2%}")

        if risk_level == "Low Risk":
            st.success(risk_level)
        elif risk_level == "Moderate Risk":
            st.warning(risk_level)
        else:
            st.error(risk_level)

        st.write(f"**Recommended Action:** {recommendation}")


with tab2:
    st.header("Model Performance")

    X = data.drop(columns=["readmitted_30_days"])
    y = data["readmitted_30_days"]

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    results_df = pd.DataFrame({
        "actual": y,
        "predicted": predictions,
        "readmission_probability": probabilities
    })

    results_df["classification_error"] = (
        results_df["actual"] != results_df["predicted"]
    ).astype(int)

    error_rate = results_df["classification_error"].mean()

    st.metric("Classification Error Rate", f"{error_rate:.2%}")
    st.dataframe(results_df.head(25), use_container_width=True)

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, predictions)

    fig, ax = plt.subplots(figsize=(6, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Readmitted", "Readmitted"]
    )
    disp.plot(ax=ax)
    ax.set_title("Readmission Prediction Confusion Matrix")
    st.pyplot(fig)

    st.subheader("Prediction Probability Distribution")

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(probabilities, bins=20)
    ax2.set_xlabel("Predicted Readmission Probability")
    ax2.set_ylabel("Number of Patients")
    ax2.set_title("Distribution of Predicted Readmission Risk")
    st.pyplot(fig2)


with tab3:
    st.header("Risk Segmentation")

    segment_col = st.selectbox(
        "Select Segment",
        ["gender", "diabetes", "hypertension", "heart_disease", "smoker", "discharge_destination"]
    )

    segment = data.groupby(segment_col).agg(
        patients=("readmitted_30_days", "count"),
        readmission_rate=("readmitted_30_days", "mean"),
        avg_age=("patient_age", "mean"),
        avg_length_of_stay=("length_of_stay", "mean"),
        avg_prior_admissions=("prior_admissions", "mean")
    ).reset_index()

    st.dataframe(segment, use_container_width=True)

    st.subheader("Readmission Rate by Segment")
    st.bar_chart(segment.set_index(segment_col)["readmission_rate"])


with tab4:
    st.header("About the Model")

    st.write("### Model Type")
    st.write("XGBoost Classifier")

    st.write("### Target")
    st.write("Predicts whether a patient will be readmitted within 30 days.")

    st.write("### Inputs")
    st.write("- Structured patient features")
    st.write("- Clinical conditions")
    st.write("- Discharge destination")
    st.write("- Physician comments")

    st.write("### Text Processing")
    st.write("Physician comments are processed using TF-IDF to identify important clinical risk words.")

    st.write("### Example Risk Words")
    st.write("- smoking")
    st.write("- shortness of breath")
    st.write("- poor medication adherence")
    st.write("- uncontrolled diabetes")
    st.write("- chest pain")
    st.write("- missed follow-up")

    st.write("### Business / Healthcare Use Case")
    st.write(
        "This type of model can support care management teams by identifying patients "
        "who may need earlier follow-up or additional discharge support."
    )


st.markdown("---")
st.caption("Built with XGBoost + TF-IDF + Streamlit | Healthcare Readmission Risk Prediction")
st.caption("Author: Feda Bashbishi fbashbis@uwaterloo.ca")