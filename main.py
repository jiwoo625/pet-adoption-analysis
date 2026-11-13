
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Pet Adoption Analysis", layout="wide")

st.title("🐶 Pet Adoption Analysis")
st.write("Analyze pet adoption trends using Machine Learning.")

uploaded_file = st.file_uploader("Upload your pet adoption CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Overview")
    st.dataframe(df.head())

    st.write("Dataset Shape:", df.shape)

    st.subheader("Data Analysis")
    st.write(df.describe(include="all"))

    if "Adoption" not in df.columns:
        st.error("The CSV file must contain an 'Adoption' column.")
        st.stop()

    # Prepare data
    data = df.copy()
    data = data.dropna(subset=["Adoption"])

    X = data.drop(columns=["Adoption"])
    y = data["Adoption"]

    # Convert categorical columns
    X = pd.get_dummies(X)

    if y.dtype == "object":
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

    if len(pd.Series(y).unique()) < 2:
        st.error("At least two adoption classes are required.")
        st.stop()

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    st.subheader("Model Performance")
    accuracy = accuracy_score(y_test, predictions)
    st.metric("Accuracy", f"{accuracy:.2%}")

    st.subheader("Feature Importance")
    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=importance.head(10), x="Importance",
                y="Feature", ax=ax)
    st.pyplot(fig)

    st.subheader("Prediction")
    st.write("Enter the pet's information to predict adoption.")

    input_data = {}
    for column in X.columns:
        input_data[column] = st.number_input(
            column, value=0.0
        )

    if st.button("Predict Adoption"):
        input_df = pd.DataFrame([input_data])
        result = model.predict(input_df)[0]

        st.success(f"Predicted adoption class: {result}")

else:
    st.info("Please upload a pet adoption CSV file to begin.")
