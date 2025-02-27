# Road Accident Severity Prediction

This project aims to predict the severity of road accidents in France using machine learning. The prediction model is built based on various factors such as weather conditions, road type, surface conditions, and time of day. The goal is to predict accident severity based on historical accident data.

## Project Structure

The project directory is organized into several key folders and files for a clean workflow:

- models/: Stores model-related files.
- notebooks/: Contains notebooks for analysis and modeling.
- reports/: For storing project-related reports.
- src/: For source code.
- requirements.txt: Lists dependencies.
- README: Overview of the project.
- references/: For citations or resources.

## Dataset
The dataset contains historical accident data from France, including key features such as:
- **Severity (grav)**: Severity of the accident
- **Location (lat, long)**: Geographical coordinates of the accident
- **Weather (atm)**: Weather conditions during the accident
- **Road type (catr)**: Type of road (e.g., autoroute, national road, etc.)
- **Surface conditions (surf)**: Road surface conditions (e.g., wet, dry)
- **Time (hrmn, mois, jour)**: Date and time of the accident

## Steps:
1. **Data Cleaning**: Preprocess the accident data by loading required columns, handling missing values, and performing feature selection.
2. **Model Training**: Train a machine learning model to predict accident severity.
3. **Evaluation**: Compare model predictions with historical accident data for evaluation.
4. **Deployment**: Deploy the model for real-time predictions using Streamlit.

## Requirements

To run this project, you’ll need the following Python libraries, which are listed in the `requirements.txt` file.
