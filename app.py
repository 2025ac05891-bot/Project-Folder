import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ML Assignment 2: Breast Cancer Classification", layout="wide")
st.title("Machine Learning Assignment 2")
st.subheader("Classification Models on Wisconsin Breast Cancer Dataset")

st.markdown("""
This app compares multiple classification models using uploaded CSV test data.  
The CSV must contain all feature columns and the **target** column.
""")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "kNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Random Forest (Ensemble)": "model/random_forest_ensemble.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}

@st.cache_resource
def load_model(path):
    return joblib.load(path)

def calculate_metrics(y_true, y_pred, y_prob=None):
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

uploaded_file = st.file_uploader("Upload test data CSV", type=["csv"])
use_sample = st.checkbox("Use included sample test_data.csv", value=True)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
elif use_sample:
    data = pd.read_csv("test_data.csv")
else:
    st.info("Please upload a CSV file or enable the included test data option.")
    st.stop()

st.write("### Uploaded / Selected Test Data")
st.dataframe(data.head(10), use_container_width=True)

if "target" not in data.columns:
    st.error("The CSV must contain a target column for evaluation.")
    st.stop()

X_test = data.drop(columns=["target"])
y_test = data["target"]

model_choice = st.selectbox("Select model", list(MODEL_FILES.keys()))
model = load_model(MODEL_FILES[model_choice])
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
metrics = calculate_metrics(y_test, y_pred, y_prob)

st.write(f"### Evaluation Metrics: {model_choice}")
metric_df = pd.DataFrame([metrics]).round(4)
st.dataframe(metric_df, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with c2:
    st.write("### Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

st.write("### Compare All Models")
all_rows = []
for name, path in MODEL_FILES.items():
    m = load_model(path)
    pred = m.predict(X_test)
    prob = m.predict_proba(X_test)[:, 1] if hasattr(m, "predict_proba") else None
    row = {"ML Model Name": name}
    row.update(calculate_metrics(y_test, pred, prob))
    all_rows.append(row)
st.dataframe(pd.DataFrame(all_rows).round(4), use_container_width=True)


import pandas as pd
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import joblib

OUTPUT_DIR = Path("model")
OUTPUT_DIR.mkdir(exist_ok=True)

data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42))]),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "kNN": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    "Support Vector Machine": Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True, random_state=42))]),
}

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    })
    filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_") + ".joblib"
    joblib.dump(model, OUTPUT_DIR / filename)

pd.DataFrame(rows).round(4).to_csv("model_metrics.csv", index=False)
pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1).to_csv("test_data.csv", index=False)
pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1).to_csv("breast_cancer_dataset.csv", index=False)
print(pd.DataFrame(rows).round(4))