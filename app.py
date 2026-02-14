import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tempfile

from model.logistic_model import get_model as logistic_model
from model.decision_tree_model import get_model as tree_model
from model.knn_model import get_model as knn_model
from model.naive_bayes_model import get_model as nb_model
from model.random_forest_model import get_model as rf_model
from model.xgboost_model import get_model as xgb_model
from model.utils import load_and_preprocess, evaluate_model

st.set_page_config(page_title="ML Assignment2", layout="centered")
st.title("Machine Learning Classification Models")

MODEL_MAP = {
    "Logistic Regression": logistic_model,
    "Decision Tree": tree_model,
    "KNN": knn_model,
    "Naive Bayes": nb_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
model_name = st.selectbox("Select Model", MODEL_MAP.keys())

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    target_column = st.selectbox(
        "Select Target Column",
        df.columns
    )
    # Save uploaded CSV temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_path = tmp.name
    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_columns,
        scaler
    ) = load_and_preprocess(temp_path, target_column)

    # Train model
    model = MODEL_MAP[model_name]()
    model.fit(X_train, y_train)

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    st.subheader("Evaluation Metrics")
    st.json(metrics)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )
    st.pyplot(fig)
