"""
Data Loading Module
This module loads and explores the IPL dataset
"""

import pandas as pd
import os
import sys

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'ipl.csv')


def load_data(filepath=DATA_PATH):
    """
    Load IPL dataset from CSV file
    
    Args:
        filepath (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    if not os.path.exists(filepath):
        print(f"❌ Error: File not found at {filepath}")
        print(f"📍 Please place your ipl.csv in: {os.path.dirname(filepath)}")
        return None
    
    print(f"✅ Loading data from: {filepath}")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"✅ Data loaded successfully!\n")
    return df


def explore_data(df):
    """
    Explore and display basic information about the dataset
    
    Args:
        df (pd.DataFrame): The dataset to explore
    """
    if df is None:
        return
    
    print("=" * 70)
    print("📊 DATASET OVERVIEW")
    print("=" * 70)
    
    # 1. Shape of data
    print(f"\n📏 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # 2. First few rows
    print(f"\n📋 First 5 rows of data:")
    print(df.head())
    
    # 3. Column information
    print(f"\n📝 Column Information:")
    print(df.info())
    
    # 4. Missing values
    print(f"\n❓ Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✅ No missing values found!")
    else:
        print(missing[missing > 0])
    
    # 5. Statistical summary
    print(f"\n📊 Statistical Summary (Numeric Columns):")
    print(df.describe())
    
    # 6. Data types
    print(f"\n🔤 Data Types:")
    print(df.dtypes)
    
    # 7. Unique values in categorical columns
    print(f"\n🏷️  Categorical Columns (Unique Values):")
    for col in df.select_dtypes(include='object').columns:
        unique_count = df[col].nunique()
        print(f"   • {col}: {unique_count} unique values")
        if unique_count <= 15:  # Show values if not too many
            print(f"     Values: {df[col].unique()}")
    
    print("\n" + "=" * 70)


def main():
    """Main function to load and explore data"""
    # Load the data
    df = load_data()
    
    # If data loaded successfully, explore it
    if df is not None:
        explore_data(df)
        return df
    
    return None


if __name__ == "__main__":
    main()
