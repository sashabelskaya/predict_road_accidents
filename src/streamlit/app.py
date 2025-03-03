import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

# Define the mapping for categories
category_mappings = {
    'season': {'Spring': 1, 'Summer': 2, 'Fall': 3, 'Winter': 4},  # For "season"
    'part_of_day': {'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4},  # For "part_of_day"
    'catu': {'Driver': 1, 'Passenger': 2, 'Pedestrian': 3},  # For "category of user"
    'lum': {'Daylight': 1, 'Twilight or dawn': 2, 'Night without public lighting': 3, 'Night with public lighting not turned on': 4, 'Night with public lighting turned on': 5},  # For "Light Conditions"
    'day_of_week': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7},  # For "Day of Week"
    'prof': {'Flat': 1, 'Slope': 2, 'Crest': 3, 'Bottom of the hill': 4},  # For "Road Profile"
    'surf': {'Normal': 1, 'Wet': 2, 'Puddles': 3, 'Flooded': 4, 'Snowy': 5, 'Muddy': 6, 'Icy': 7, 'Grease': 8, 'Other': 9}  # For "Road Surface"
}

# Load the trained model and encoder
MODEL_PATH = "./output/xgb_model.pkl"
SCALER_PATH = "./output/scaler.pkl"
FEATURES_PATH = "./output/feature_names.pkl"
OUTPUT_DIR = "/Users/alexandra/Desktop/france/NOV24_BDS_INT_Accidents/output"
METRICS_PATH = os.path.join(OUTPUT_DIR, "model_metrics.json")

# Check if all required files exist
missing_files = [path for path in [MODEL_PATH, SCALER_PATH, FEATURES_PATH] if not os.path.exists(path)]
if missing_files:
    st.error(f"Missing required files: {', '.join(missing_files)}")
    st.stop()

# Load trained model, encoder, and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
expected_features = joblib.load(FEATURES_PATH)

# Custom CSS to change the background and navigation styles
st.markdown("""
    <style>
        body {
            background-color: #ADD8E6;  /* Light Blue */
            font-family: 'Arial', sans-serif;
        }
        .main {
            background-color: #ADD8E6;  /* Light Blue for main container */
        }
        .css-1d391kg {
            background-color: #4CAF50; 
            color: white;
        }
        .css-1v3fvcr {
            background-color: #333;
            color: white;
        }
        .sidebar .sidebar-content {
            background-color: #333;
            color: white;
        }
        .css-18e3th9 {
            background-color: #4CAF50;
        }
        .css-1g4c0o0 {
            font-size: 18px;
            font-weight: bold;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", ("Home", "Prediction", "Visualizations", "Model Exploration"))

# Home Page

def show_home_page():
    st.title("Predicting Road Accident Severity in France")

    st.header("Introduction to the Project")
    st.subheader("Context")
    st.write("""
    Road accidents are a significant concern for public safety in France. Predicting accident severity can help improve emergency response, optimize road safety measures, and influence infrastructure planning.
    """)

    st.subheader("Integration into Business")
    st.write("""
    By predicting the severity of accidents, this project aids public safety efforts and supports traffic management, emergency preparedness, and infrastructure investments. With accurate predictions, relevant authorities can make informed decisions to improve safety and reduce overall accident severity.
    """)

    st.subheader("Technical Perspective")
    st.write("""
    This project involves processing a large dataset using machine learning techniques to understand the relationship between various factors and accident severity. Data preprocessing steps, including handling missing values, feature engineering, and normalization, are essential components.
    """)

    st.subheader("Economic Perspective")
    st.write("""
    Reducing accident severity has far-reaching economic benefits. It can lead to cost savings in healthcare, insurance premiums, and infrastructure repairs, contributing to the financial well-being of the public and government.
    """)

    st.subheader("Scientific Perspective")
    st.write("""
    Machine learning techniques are applied to assess risk factors in road safety, enabling the creation of risk-scoring models based on environmental and road-related features. The insights can drive research and improvements in traffic safety.
    """)

    st.header("Project Objectives")
    st.write("""
    The primary goal of this project is to:
    - Develop a predictive model to assess the severity of accidents.
    """)

    st.header("Understanding and Manipulation of Data")
    st.subheader("Framework")
    st.write("""
    **Dataset Source:** Data from [gov.fr](https://www.data.gouv.fr) (2019-2023)
    *Note: Data from 2019 and onwards cannot be directly compared with previous years due to changes in labeling.*
    
    **Availability:** The dataset is publicly available, with specific warnings on data labeling changes after 2019.
    
    **Volume:** A large dataset covering metropolitan France, containing crucial details on accident scenarios.
    """)

    st.subheader("Relevance")
    st.write("""
    **Key Features:** Weather conditions, road surface type, accident severity, and time-based attributes like season, day of the week, and time of day.
    
    **Target Variable:** The severity of accidents ('grav'), classified into 4 categories:
    - 0: No injury
    - 1: Minor injury
    - 2: Hospitalized
    - 3: Killed
    """)

    st.subheader("Pre-Processing")
    st.write("""
    - **Data Cleaning:** Standardized feature names, handled missing values, and prepared the dataset for analysis.
    - **Transformation:** After splitting the data, normalization was applied to scale the features, ensuring that they are suitable for the machine learning model.
    """)

    st.header("Next Steps")
    st.write("""
    - Understanding how different features influence accident outcomes.
    - Predicting accident severity based on specific parameters.

    Stay tuned for more details in the other sections of the app!
    """)

# Prediction Page
def show_prediction():
    st.title("Accident Severity Prediction")
    
    # User input fields (dropdown lists updated to show strings and map to numerical values)
    season = st.selectbox("Season", list(category_mappings['season'].keys()))
    part_of_day = st.selectbox("Part of Day", list(category_mappings['part_of_day'].keys()))
    catu = st.selectbox("Category of user", list(category_mappings['catu'].keys()))
    lum = st.selectbox("Light Conditions", list(category_mappings['lum'].keys()))
    day_of_week = st.selectbox("Day of Week", list(category_mappings['day_of_week'].keys()))
    prof = st.selectbox("Road Profile", list(category_mappings['prof'].keys()))
    surf = st.selectbox("Road Surface", list(category_mappings['surf'].keys()))

    # Convert user input to DataFrame, using the corresponding numeric values from the category mappings
    input_data = pd.DataFrame({
        "season": [category_mappings['season'][season]],
        "part_of_day": [category_mappings['part_of_day'][part_of_day]],
        "catu": [category_mappings['catu'][catu]],
        "lum": [category_mappings['lum'][lum]],
        "day_of_week": [category_mappings['day_of_week'][day_of_week]],
        "prof": [category_mappings['prof'][prof]],
        "surf": [category_mappings['surf'][surf]]
    })

    # Ensure the feature order is consistent with the trained model
    input_data = input_data.reindex(columns=expected_features, fill_value=0)

    # Apply scaling before prediction
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    if st.button("Predict"):
        prediction = model.predict(input_data_scaled)
        st.success(f"Predicted Severity: {prediction[0]}")

# Visualizations Page
def show_visualizations():
    st.title("Data Visualizations")
    
    # Load the dataset to display visualizations
    df = pd.read_csv('./data/preprocessed_data.csv')
    
    # Plot a correlation matrix for the dataset
    st.subheader("Correlation Matrix")
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    
    # Plot feature importance (from the model)
    st.subheader("Feature Importance")
    feature_importances = model.feature_importances_
    features = expected_features  # List of feature names
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
    st.pyplot(fig)
    
    # Add the mapping for 'grav' categories
    st.subheader("Distribution of 'grav' Category")
    if 'grav' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(data=df, x='grav', ax=ax)
        ax.set_title("Distribution of 'grav' Category (Accident Severity)")
        ax.set_xlabel('Accident Severity (grav)')
        ax.set_ylabel('Count')
        st.pyplot(fig)
    else:
        st.warning("The 'grav' column is not found in the dataset.")

    # Add the mapping for 'catu' categories
    catu_mapping = {
        1: 'Driver',
        2: 'Passenger',
        3: 'Pedestrian'
    }
    grav_mapping = {
        0: 'No injury',
        1: 'Minor injury',
        2: 'Hospitalized',
        3: 'Killed'
    }

    # Visualize the distribution of 'catu' category
    st.subheader("Distribution of 'catu' Category (Category of User)")
    if 'catu' in df.columns:
        # Map the 'catu' column to the defined categories
        df['catu_mapped'] = df['catu'].map(catu_mapping)

        # Plot the distribution of 'catu'
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(data=df, x='catu_mapped', ax=ax)
        ax.set_title("Distribution of 'catu' Category (Category of User)")
        ax.set_xlabel('Category of User (catu)')
        ax.set_ylabel('Count')
        st.pyplot(fig)
    else:
        st.warning("The 'catu' column is not found in the dataset.")

    # Visualize 'catu' against 'grav' (accident severity)
    st.subheader("Category of User (catu) vs Accident Severity (grav)")
    if 'catu' in df.columns and 'grav' in df.columns:
        # Map 'grav' column to readable categories
        df['grav_mapped'] = df['grav'].map(grav_mapping)
        
        # Plot the relationship between 'catu' and 'grav'
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(data=df, x='catu_mapped', hue='grav_mapped', ax=ax)
        ax.set_title("Category of User (catu) vs Accident Severity (grav)")
        ax.set_xlabel('Category of User')
        ax.set_ylabel('Count')
        st.pyplot(fig)
    else:
        st.warning("The 'catu' and 'grav' columns are not found in the dataset.")

    # Visualize 'day_of_week' against 'grav' (accident severity)
    st.subheader("Day of Week vs Accident Severity (grav)")
    if 'day_of_week' in df.columns and 'grav' in df.columns:
        # Map 'grav' column to readable categories
        df['grav_mapped'] = df['grav'].map(grav_mapping)
        
        # Plot the relationship between 'day_of_week' and 'grav'
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(data=df, x='day_of_week', hue='grav_mapped', ax=ax)
        ax.set_title("Day of Week vs Accident Severity (grav)")
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Count')
        st.pyplot(fig)
    else:
        st.warning("The 'day_of_week' and 'grav' columns are not found in the dataset.")

# Model Exploration Page (Continued)
def show_model_insights():
    st.title("Model Insights and Explanation")

    # Load saved metrics
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)

        # Show Model Performance
        st.subheader("Model Performance")
        st.write(f"ROC AUC Score: {metrics['roc_auc']:.4f}")

        # Show Classification Report
        st.subheader("Classification Report")
        st.markdown("""The classification report provides metrics such as precision, recall, and f1-score for each class, along with accuracy, macro average, and weighted average for the entire model.""")
        st.json(metrics["classification_report"])  # Display full report in JSON format
        
        # Show Confusion Matrix
        st.subheader("Confusion Matrix")
        st.markdown("""
            The confusion matrix shows the true vs. predicted values, helping to assess the performance of the model.
        """)
        cm = np.array(metrics["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

    else:
        st.warning("Model metrics not found. Please train the model first.")

# Display selected page content
if page == "Home":
    show_home_page()
elif page == "Prediction":
    show_prediction()
elif page == "Visualizations":
    show_visualizations()
elif page == "Model Exploration":
    show_model_insights()
