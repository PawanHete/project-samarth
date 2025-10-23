import pandas as pd
import numpy as np
import os # Import the os module to handle file paths robustly

# --- Define Correct File Paths ---
# This tells the script to look inside the correct sub-folders.
RAW_DATA_DIR = 'data/raw/'
PROCESSED_DATA_DIR = 'data/processed/'

# Construct the full path for each file
CROP_DATA_FILE = os.path.join(RAW_DATA_DIR, 'State and Disrtrict wise Crop Production.csv')
RAIN_DATA_FILE = os.path.join(RAW_DATA_DIR, 'district wise rainfall normal.csv')

CLEANED_CROP_FILE = os.path.join(PROCESSED_DATA_DIR, 'crop_production_cleaned.csv')
CLEANED_RAIN_FILE = os.path.join(PROCESSED_DATA_DIR, 'rainfall_cleaned.csv')


# --- Load the Datasets ---
try:
    # Use the full paths we just defined
    crop_df = pd.read_csv(CROP_DATA_FILE)
    rain_df = pd.read_csv(RAIN_DATA_FILE)
    print("✅ Datasets loaded successfully from 'data/raw/' folder!")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    print("Please make sure your raw data files are inside the 'data/raw/' folder.")
    exit()

# --- 1. Clean the Crop Production Data ---
print("\nCleaning Crop Production data...")

# Handle missing Production values
initial_missing = crop_df['Production'].isnull().sum()
crop_df['Production'].fillna(0, inplace=True)
print(f"   - Handled {initial_missing} missing values in 'Production' column.")

# Standardize text columns
text_cols = ['State_Name', 'District_Name', 'Season', 'Crop']
for col in text_cols:
    crop_df[col] = crop_df[col].str.strip().str.upper()
print("   - Standardized text columns to uppercase and stripped whitespace.")

# Feature Engineering: Calculate Yield
# Use np.divide for safe division to avoid errors when Area is 0
crop_df['Yield'] = np.divide(crop_df['Production'], crop_df['Area'])
# Replace any infinite values (from division by zero) with 0
crop_df.replace([np.inf, -np.inf], 0, inplace=True)
crop_df['Yield'].fillna(0, inplace=True)
print("   - Engineered 'Yield' feature (Production / Area).")


# --- 2. Clean the Rainfall Data ---
print("\nCleaning Rainfall data...")

# Rename columns for consistency
rain_df.rename(columns={'STATE_UT_NAME': 'State_Name', 'DISTRICT': 'District_Name'}, inplace=True)
print("   - Renamed columns to match crop data.")

# Standardize text columns
rain_df['State_Name'] = rain_df['State_Name'].str.strip().str.upper()
rain_df['District_Name'] = rain_df['District_Name'].str.strip().str.upper()
print("   - Standardized text columns.")


# --- 3. Save the Cleaned Data ---
try:
    # Use the full paths for saving the cleaned files
    crop_df.to_csv(CLEANED_CROP_FILE, index=False)
    rain_df.to_csv(CLEANED_RAIN_FILE, index=False)
    print("\n✅ Successfully saved cleaned data to the 'data/processed/' folder.")
except Exception as e:
    print(f"\n❌ Error saving files: {e}")

# --- Display a sample of the cleaned data ---
print("\n--- Sample of Cleaned Crop Data ---")
print(crop_df.head())
print("\n--- Sample of Cleaned Rainfall Data ---")
print(rain_df.head())
