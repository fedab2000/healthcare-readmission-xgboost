import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def train_readmission_model():
    df = pd.read_csv("data/healthcare_readmission_dataset.csv")

    X = df.drop(columns=["readmitted_30_days"])
    y = df["readmitted_30_days"]

    text_col = "physician_comments"
    categorical_cols = ["gender", "discharge_destination"]

    numerical_cols = [
        col for col in X.columns
        if col not in categorical_cols + [text_col]
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(drop="first"), categorical_cols),
            ("text", TfidfVectorizer(
                max_features=100,
                stop_words="english",
                ngram_range=(1, 2)
            ), text_col)
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n=== XGBoost Readmission Model Evaluation ===")
    print(classification_report(y_test, predictions, zero_division=0))
    print("ROC AUC:", round(roc_auc_score(y_test, probabilities), 3))

    os.makedirs("outputs", exist_ok=True)

    # Save model
    joblib.dump(model, "outputs/readmission_xgboost_model.joblib")

    # Save prediction errors
    results_df = pd.DataFrame({
        "actual": y_test.values,
        "predicted": predictions,
        "readmission_probability": probabilities
    })

    results_df["classification_error"] = (
        results_df["actual"] != results_df["predicted"]
    ).astype(int)

    results_df["probability_error"] = abs(
        results_df["actual"] - results_df["readmission_probability"]
    )

    results_df.to_csv(
        "outputs/readmission_prediction_errors.csv",
        index=False
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Readmitted", "Readmitted"]
    )

    disp.plot()
    plt.title("Readmission Prediction Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/readmission_confusion_matrix.png")
    plt.show()

    # Error histogram
    plt.figure(figsize=(8, 5))
    plt.hist(results_df["probability_error"], bins=20)
    plt.xlabel("Prediction Probability Error")
    plt.ylabel("Number of Patients")
    plt.title("Distribution of Readmission Prediction Errors")
    plt.tight_layout()
    plt.savefig("outputs/readmission_error_histogram.png")
    plt.show()

    print("\n✅ Model saved to outputs/readmission_xgboost_model.joblib")
    print("✅ Prediction errors saved to outputs/readmission_prediction_errors.csv")
    print("✅ Confusion matrix saved to outputs/readmission_confusion_matrix.png")
    print("✅ Error histogram saved to outputs/readmission_error_histogram.png")

    return model


if __name__ == "__main__":
    train_readmission_model()