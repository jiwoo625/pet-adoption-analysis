
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Pet Adoption Analysis", layout="wide")

st.title("🐶 Pet Adoption Analysis")
st.write("Analyze pet adoption trends using Machine Learning.")

# Automatically load CSV from GitHub repository
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "pet_adoption_dataset.csv")

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)

    st.success("CSV dataset loaded successfully!")

    st.subheader("Dataset Overview")
    st.dataframe(df)

    st.subheader("Adoption Statistics")
    st.bar_chart(df["Adoption"].value_counts())

    st.subheader("Age Analysis")
    st.bar_chart(df.groupby("Age")["Stay_Days"].mean())

    # Prepare data
    data = df.dropna(subset=["Adoption"]).copy()
    X = pd.get_dummies(
        data[["Animal_Type", "Age", "Stay_Days", "Health_Status"]]
    )
    y = data["Adoption"]

    if y.nunique() < 2:
        st.error("At least two adoption classes are required.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = RandomForestClassifier(
            n_estimators=100, random_state=42
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        st.subheader("Model Performance")
        st.metric("Accuracy", f"{accuracy:.2%}")

        st.subheader("Feature Importance")
        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        st.bar_chart(importance.set_index("Feature"))

else:
    st.error("CSV file not found!")
    st.write("Expected file:", DATA_PATH)
    st.write("Files in project folder:", os.listdir(BASE_DIR))
