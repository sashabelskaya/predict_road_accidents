import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.preprocessing import OneHotEncoder
import joblib

# Kaggle dataset
KAGGLE_DATASET = "alexandrabelskaya/metropolitan-france"
DATA_DIR = "data"
DATA_FILE = "metropolitan_france_df_cleaned.csv"
DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)


os.makedirs(DATA_DIR, exist_ok=True)

# Download if it doesn't exist
if not os.path.exists(DATA_PATH):
    print("Downloading dataset from Kaggle...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=DATA_DIR, unzip=True)
    print("Download complete!")

# Load 
df = pd.read_csv(DATA_PATH)
print("Dataset Loaded Successfully!")
print(df.head())  # Show preview

# Print the column names
print("Column Names in the Dataset:")
print(df.columns)

# Save the  encoder
def save_encoder(one_hot):
    output_dir = "output" 
    os.makedirs(output_dir, exist_ok=True)
    encoder_path = os.path.join(output_dir, "encoder.pkl")
    joblib.dump(one_hot, encoder_path)
    print(f"OneHotEncoder saved to '{encoder_path}'")

# Preprocessing function
def load_and_preprocess_data(df):
    df.columns = df.columns.str.lower()
    
    categorical_columns = ['season', 'part_of_day']
    one_hot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # Ignore unknown categories during transformation
    
    encoded_features = pd.DataFrame(
        one_hot.fit_transform(df[categorical_columns]), 
        columns=one_hot.get_feature_names_out(categorical_columns)  
    )
    
    save_encoder(one_hot)

    df = df.drop(columns=categorical_columns).join(encoded_features)

    print("Columns after one-hot encoding:")
    print(df.columns)

    columns_to_keep = ['catu', 'lum', 'day_of_week', 'prof', 'surf'] + list(encoded_features.columns) + ['grav']
    
    simplified_df = df.loc[:, columns_to_keep].copy()

    grav_mapping = {4: 0, 1: 1, 3: 2, 2: 3}  # Map original severity levels to new values (e.g., binary or ordinal labels)
    simplified_df['grav'] = simplified_df['grav'].map(grav_mapping)

    return simplified_df

simplified_df = load_and_preprocess_data(df)

print("Preprocessing Complete!")
print(simplified_df.head())

preprocessed_data_path = os.path.join(DATA_DIR, "preprocessed_data.csv")
simplified_df.to_csv(preprocessed_data_path, index=False)
print(f"Preprocessed data saved to '{preprocessed_data_path}'")
