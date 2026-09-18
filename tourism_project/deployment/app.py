
import streamlit as st
import pandas as pd
import joblib
import os

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction")

st.write(
    "Enter the customer's information below to predict "
    "whether the customer is likely to purchase the tourism package."
)

# Customer inputs
Age = st.number_input("Age", min_value=18, max_value=100, value=30)

TypeofContact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

CityTier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

DurationOfPitch = st.number_input(
    "Duration of Pitch",
    min_value=0,
    value=10
)

Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Free Lancer", "Small Business", "Large Business"]
)

Gender = st.selectbox(
    "Gender",
    ["Male", "Female", "Others"]
)

NumberOfPersonVisiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    value=2
)

NumberOfFollowups = st.number_input(
    "Number of Follow-ups",
    min_value=0,
    value=3
)

ProductPitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
)

PreferredPropertyStar = st.selectbox(
    "Preferred Property Star",
    [3, 4, 5]
)

MaritalStatus = st.selectbox(
    "Marital Status",
    ["Married", "Single", "Divorced", "Unmarried"]
)

NumberOfTrips = st.number_input(
    "Number of Trips",
    min_value=0,
    value=2
)

Passport = st.selectbox(
    "Passport",
    ["Yes", "No"]
)

PitchSatisfactionScore = st.selectbox(
    "Pitch Satisfaction Score",
    [1, 2, 3, 4, 5]
)

OwnCar = st.selectbox(
    "Own Car",
    ["Yes", "No"]
)

NumberOfChildrenVisiting = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    value=0
)

Designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

MonthlyIncome = st.number_input(
    "Monthly Income",
    min_value=0.0,
    value=25000.0
)

# Convert Yes/No values to 1/0
Passport = 1 if Passport == "Yes" else 0
OwnCar = 1 if OwnCar == "Yes" else 0

# Create input dataframe
input_data = pd.DataFrame({
    "Age": [Age],
    "TypeofContact": [TypeofContact],
    "CityTier": [CityTier],
    "DurationOfPitch": [DurationOfPitch],
    "Occupation": [Occupation],
    "Gender": [Gender],
    "NumberOfPersonVisiting": [NumberOfPersonVisiting],
    "NumberOfFollowups": [NumberOfFollowups],
    "ProductPitched": [ProductPitched],
    "PreferredPropertyStar": [PreferredPropertyStar],
    "MaritalStatus": [MaritalStatus],
    "NumberOfTrips": [NumberOfTrips],
    "Passport": [Passport],
    "PitchSatisfactionScore": [PitchSatisfactionScore],
    "OwnCar": [OwnCar],
    "NumberOfChildrenVisiting": [NumberOfChildrenVisiting],
    "Designation": [Designation],
    "MonthlyIncome": [MonthlyIncome]
})

# Prediction
if st.button("Predict Tourism Package Purchase"):

    probability = model.predict_proba(input_data)[0][1]

    classification_threshold = 0.45

    prediction = 1 if probability >= classification_threshold else 0

    if prediction == 1:
        st.success(
            "The customer is predicted to purchase the tourism package."
        )
    else:
        st.warning(
            "The customer is predicted not to purchase the tourism package."
        )

    st.metric(
        "Purchase Probability",
        f"{probability * 100:.2f}%"
    )
