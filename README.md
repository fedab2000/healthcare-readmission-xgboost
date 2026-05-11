# healthcare-readmission-xgboost
# Healthcare Readmission Risk Prediction using XGBoost + NLP

## Overview

This project predicts whether a patient is likely to be readmitted to the hospital within 30 days using:

- Structured healthcare data
- Physician clinical notes
- XGBoost machine learning
- TF-IDF natural language processing (NLP)
- Streamlit interactive dashboard

The project demonstrates how machine learning and NLP can support healthcare risk prediction and care management workflows.

---

# Features

## Machine Learning
- XGBoost Classifier
- Binary classification for 30-day readmission prediction
- Structured + unstructured data modeling

## NLP
- TF-IDF vectorization of physician comments
- Important clinical risk words influence predictions

## Dashboard
- Patient risk calculator
- Model performance analysis
- Readmission risk segmentation
- Confusion matrix visualization
- Clinical recommendation generation

---

# Technologies Used

- Python
- Scikit-learn
- XGBoost
- Pandas
- Streamlit
- Matplotlib
- TF-IDF NLP

---

# Project Structure

```text
healthcare-readmission-xgboost/
│
├── data/
│   └── healthcare_readmission_dataset.csv
│
├── outputs/
│   ├── readmission_xgboost_model.joblib
│   ├── readmission_prediction_errors.csv
│   ├── readmission_confusion_matrix.png
│   └── readmission_error_histogram.png
│
├── generate_healthcare_dataset.py
├── train_readmission_model.py
├── streamlit_app.py
├── requirements.txt
└── README.md
Dataset

The synthetic healthcare dataset includes:

Feature	              Description
patient_age	          Patient age
gender	              Male/Female
length_of_stay	      Hospital stay duration
prior_admissions	    Previous hospital admissions
emergency_visits	    Recent emergency visits
diabetes	            Diabetes condition flag
hypertension	        Hypertension flag
heart_disease	        Heart disease flag
smoker	              Smoking status
bmi	Body              Mass Index
medication_count	    Number of medications
discharge_destination	Discharge location
physician_comments	  Physician clinical notes
readmitted_30_days	  Target variable

Physician Comments + NLP

The project includes synthetic physician notes such as:

Patient continues smoking and reports shortness of breath.
Poor medication adherence and missed recent follow-up appointment.
Uncontrolled diabetes with elevated glucose levels.

TF-IDF is used to identify important risk-related words and phrases.

Machine Learning Pipeline
Structured Data
StandardScaler
OneHotEncoder
Text Data
TF-IDF Vectorizer
Model
XGBoost Classifier

Evaluation Metrics

The model includes:

Accuracy
Precision
Recall
F1-score
ROC AUC
Confusion Matrix
Prediction error analysis

Installation
Clone Repository
git clone https://github.com/YOUR_USERNAME/healthcare-readmission-xgboost.git
cd healthcare-readmission-xgboost
Install Requirements
pip install -r requirements.txt

Generate Dataset
python generate_healthcare_dataset.py
Train Model
python train_readmission_model.py
Run Streamlit App
streamlit run streamlit_app.py

Example Application Workflow
Enter patient clinical information
Enter physician comments
Model predicts:
Readmission probability
Risk level
Recommended action

Example output:

Readmission Probability: 82%
Risk Level: High Risk
Recommended Action:
Recommend early follow-up and care management review.

Example Healthcare Use Cases
Hospital readmission prevention
Care management prioritization
Clinical decision support
Patient risk stratification
Population health analytics
Future Enhancements

Potential future improvements:

SHAP explainability
Deep learning NLP models
Real healthcare datasets
Time-series patient monitoring
LLM-powered physician note summarization
Care recommendation engine

Author

Feda Bashbishi

Enterprise Data Analytics & AI Leader
University of Waterloo MDSAI Candidate
