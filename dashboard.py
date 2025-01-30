import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Set page configuration
st.set_page_config(
    page_title="Titanic Survival Dashboard", page_icon="🚢", layout="wide"
)


# Load the data
@st.cache_data
def load_data():
    # Using the direct URL to the Titanic dataset from seaborn
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    df = pd.read_csv(url)
    return df


df = load_data()

# Title
st.title("🚢 Titanic Survival Analysis Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to", ["Dataset Overview", "Visualizations", "Prediction Model"]
)

if page == "Dataset Overview":
    st.header("Dataset Overview")

    # Display basic information
    st.subheader("First few rows of the dataset")
    st.dataframe(df.head())

    st.subheader("Dataset Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Dataset Shape:", df.shape)
    with col2:
        st.write("Survival Rate:", f"{(df['Survived'].mean() * 100):.2f}%")

    # Display missing values
    st.subheader("Missing Values")
    missing_values = df.isnull().sum()
    st.write(missing_values[missing_values > 0])

elif page == "Visualizations":
    st.header("Data Visualizations")

    # Survival by Gender
    st.subheader("Survival Rate by Gender")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x="Sex", y="Survived")
    st.pyplot(fig1)

    # Survival by Passenger Class
    st.subheader("Survival Rate by Passenger Class")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x="Pclass", y="Survived")
    st.pyplot(fig2)

    # Age Distribution
    st.subheader("Age Distribution by Survival")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x="Age", hue="Survived", multiple="stack")
    st.pyplot(fig3)

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation = df[numeric_cols].corr()
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", center=0)
    st.pyplot(fig4)

else:  # Prediction Model
    st.header("Survival Prediction Model")

    # Prepare the data
    def prepare_data(df):
        # Create a copy of the dataframe to avoid modifying the original
        df = df.copy()

        # Handle missing values for all relevant columns
        df["Age"].fillna(df["Age"].median(), inplace=True)
        df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
        df["Fare"].fillna(df["Fare"].median(), inplace=True)

        # Convert categorical variables
        df = pd.get_dummies(df, columns=["Sex", "Embarked"])

        # Select features
        features = [
            "Pclass",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Sex_female",
            "Sex_male",
            "Embarked_C",
            "Embarked_Q",
            "Embarked_S",
        ]

        # Ensure all features exist in the dataframe
        for feature in features:
            if feature not in df.columns:
                df[feature] = 0

        return df[features]

    X = prepare_data(df)
    y = df["Survived"]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Model performance
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.subheader("Model Performance")
    st.write(f"Model Accuracy: {accuracy:.2%}")

    # Feature importance
    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame(
        {"Feature": X.columns, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)

    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feature_importance, x="Importance", y="Feature")
    plt.title("Feature Importance in Survival Prediction")
    st.pyplot(fig5)

    # Interactive Prediction
    st.subheader("Make a Prediction")

    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3])
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.slider("Age", 0, 100, 30)

    with col2:
        sibsp = st.number_input("Number of Siblings/Spouses", 0, 10, 0)
        parch = st.number_input("Number of Parents/Children", 0, 10, 0)
        embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

    # Create a sample passenger
    sample_passenger = pd.DataFrame(
        {
            "Pclass": [pclass],
            "Age": [age],
            "SibSp": [sibsp],
            "Parch": [parch],
            "Fare": [df["Fare"].mean()],  # Using mean fare for simplicity
            "Sex_female": [1 if sex == "female" else 0],
            "Sex_male": [1 if sex == "male" else 0],
            "Embarked_C": [1 if embarked == "C" else 0],
            "Embarked_Q": [1 if embarked == "Q" else 0],
            "Embarked_S": [1 if embarked == "S" else 0],
        }
    )

    if st.button("Predict Survival"):
        prediction = model.predict(sample_passenger)
        probability = model.predict_proba(sample_passenger)

        st.write("---")
        if prediction[0] == 1:
            st.success(
                f"This passenger would likely SURVIVE with {probability[0][1]:.1%} probability"
            )
        else:
            st.error(
                f"This passenger would likely NOT SURVIVE with {probability[0][0]:.1%} probability"
            )
