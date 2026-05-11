import os
import random
import numpy as np
import pandas as pd

np.random.seed(42)
random.seed(42)

n = 5000

# -----------------------------
# Structured patient data
# -----------------------------
patient_age = np.random.randint(18, 90, n)

gender = np.random.choice(
    ["Female", "Male"],
    size=n,
    p=[0.52, 0.48]
)

length_of_stay = np.random.poisson(4, n).clip(1, 30)
prior_admissions = np.random.poisson(0.8, n)
emergency_visits = np.random.poisson(0.6, n)

diabetes = np.random.binomial(1, 0.28, n)
hypertension = np.random.binomial(1, 0.35, n)
heart_disease = np.random.binomial(1, 0.22, n)
smoker = np.random.binomial(1, 0.24, n)

bmi = np.random.normal(28, 6, n).clip(16, 55).round(1)
medication_count = np.random.poisson(4, n).clip(0, 20)

discharge_destination = np.random.choice(
    ["Home", "Home with Support", "Rehab Facility", "Long-Term Care"],
    size=n,
    p=[0.58, 0.25, 0.12, 0.05]
)

# -----------------------------
# Physician comment templates
# -----------------------------
high_risk_comments = [
    "Patient continues smoking and reports shortness of breath.",
    "Poor medication adherence and missed recent follow-up appointment.",
    "Uncontrolled diabetes with elevated glucose levels.",
    "Patient reports frequent chest pain and fatigue.",
    "High sodium diet and worsening heart failure symptoms.",
    "Severe obesity and limited physical activity.",
    "Patient eats processed meat frequently and reports poor diet quality.",
    "Multiple recent emergency visits and unstable symptoms.",
    "Patient lives alone and has limited home support.",
    "Ongoing hypertension concerns and inconsistent medication use."
]

moderate_risk_comments = [
    "Patient has some medication adherence concerns but follow-up is scheduled.",
    "Mild symptoms reported with moderate lifestyle risk factors.",
    "Patient needs diet counseling and closer monitoring.",
    "Blood pressure remains slightly elevated.",
    "Patient reports occasional shortness of breath during activity.",
    "Some missed appointments noted in the last few months.",
    "Patient has limited exercise and moderate weight concerns."
]

low_risk_comments = [
    "Patient is stable and compliant with medications.",
    "Improved condition with follow-up appointment scheduled.",
    "Blood pressure is controlled and symptoms are stable.",
    "Patient reports regular exercise and improved diet.",
    "No acute concerns noted during discharge planning.",
    "Patient has strong family support and understands care plan.",
    "Medication adherence is good and condition appears stable."
]

comments = []
text_risk_score = []

for i in range(n):
    risk_points = 0

    if smoker[i] == 1:
        risk_points += 1
    if diabetes[i] == 1:
        risk_points += 1
    if heart_disease[i] == 1:
        risk_points += 1
    if prior_admissions[i] >= 2:
        risk_points += 1
    if emergency_visits[i] >= 2:
        risk_points += 1
    if bmi[i] >= 35:
        risk_points += 1
    if discharge_destination[i] in ["Rehab Facility", "Long-Term Care"]:
        risk_points += 1

    if risk_points >= 4:
        comment = random.choice(high_risk_comments)
        text_score = 1.2
    elif risk_points >= 2:
        comment = random.choice(moderate_risk_comments)
        text_score = 0.5
    else:
        comment = random.choice(low_risk_comments)
        text_score = -0.6

    comments.append(comment)
    text_risk_score.append(text_score)

text_risk_score = np.array(text_risk_score)

# -----------------------------
# Readmission probability logic
# -----------------------------
risk_score = (
    -3.2
    + 0.025 * patient_age
    + 0.18 * length_of_stay
    + 0.45 * prior_admissions
    + 0.35 * emergency_visits
    + 0.55 * diabetes
    + 0.35 * hypertension
    + 0.75 * heart_disease
    + 0.45 * smoker
    + 0.04 * medication_count
    + 0.035 * bmi
    + text_risk_score
)

risk_score += np.where(discharge_destination == "Home with Support", 0.25, 0)
risk_score += np.where(discharge_destination == "Rehab Facility", 0.55, 0)
risk_score += np.where(discharge_destination == "Long-Term Care", 0.85, 0)

readmission_probability = 1 / (1 + np.exp(-risk_score))
readmitted_30_days = np.random.binomial(1, readmission_probability)

# -----------------------------
# Final dataset
# -----------------------------
df = pd.DataFrame({
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
    "physician_comments": comments,
    "readmitted_30_days": readmitted_30_days
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/healthcare_readmission_dataset.csv", index=False)

print("Dataset created: data/healthcare_readmission_dataset.csv")
print(df.head())
print("\nReadmission rate:", round(df["readmitted_30_days"].mean(), 3))