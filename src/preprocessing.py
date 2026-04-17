"""
Data Preprocessing Module
This module cleans and prepares data for machine learning
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import sys

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'ipl.csv')


def load_data(filepath=DATA_PATH):
    """Load the IPL dataset"""
    if not os.path.exists(filepath):
        print(f"❌ Error: File not found at {filepath}")
        return None
    
    df = pd.read_csv(filepath, low_memory=False)
    print(f"✅ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns\n")
    return df


def build_match_summary(df):
    """
    Convert ball-by-ball IPL data into one row per match.

    Some IPL CSV files contain every ball of every match. In that case,
    we only need one row per match for prediction.
    """
    if {'match_id', 'innings', 'batting_team', 'bowling_team', 'match_won_by'}.issubset(df.columns):
        print("🔧 Building match-level summary from ball-by-ball dataset")
        match_cols = ['match_id', 'batting_team', 'bowling_team', 'match_won_by']
        for optional in ['venue', 'city', 'toss_winner', 'toss_decision', 'season', 'stage']:
            if optional in df.columns:
                match_cols.append(optional)
        summary = df[df['innings'] == 1][match_cols].drop_duplicates(subset=['match_id'])
        summary = summary.rename(columns={
            'batting_team': 'team1',
            'bowling_team': 'team2',
            'match_won_by': 'winner'
        })
        print(f"✅ Match summary created: {summary.shape[0]} matches\n")
        return summary
    return df


def select_important_columns(df):
    """
    Select only the columns we need for prediction.
    """
    print("🔍 STEP 1: Selecting Important Columns")
    print("-" * 60)
    
    # Common column names in IPL datasets
    possible_team1_names = ['team1', 'Team1', 'batting_team', 'Batting_Team']
    possible_team2_names = ['team2', 'Team2', 'bowling_team', 'Bowling_Team']
    possible_winner_names = ['winner', 'Winner', 'Match_Winner', 'match_winner', 'match_won_by']
    possible_venue_names = ['venue', 'Venue', 'ground', 'Ground']
    possible_city_names = ['city', 'City']
    
    # Find which columns exist in the dataset
    team1_col = next((col for col in possible_team1_names if col in df.columns), None)
    team2_col = next((col for col in possible_team2_names if col in df.columns), None)
    winner_col = next((col for col in possible_winner_names if col in df.columns), None)
    venue_col = next((col for col in possible_venue_names if col in df.columns), None)
    city_col = next((col for col in possible_city_names if col in df.columns), None)
    
    # Check if critical columns exist
    if not all([team1_col, team2_col, winner_col]):
        print("⚠️  Warning: Expected columns not found exactly.")
        print(f"   Available columns: {df.columns.tolist()}")
        print("   Please check your dataset structure.\n")
        print("   Looking for: team1, team2, winner columns")
        return None
    
    print(f"✅ Found columns:")
    print(f"   • Team 1: {team1_col}")
    print(f"   • Team 2: {team2_col}")
    print(f"   • Winner: {winner_col}")
    if venue_col:
        print(f"   • Venue: {venue_col}")
    if city_col:
        print(f"   • City: {city_col}")
    
    selected_columns = [team1_col, team2_col, winner_col]
    if venue_col:
        selected_columns.append(venue_col)
    if city_col:
        selected_columns.append(city_col)
    
    df_selected = df[selected_columns].copy()
    
    rename_dict = {
        team1_col: 'team1',
        team2_col: 'team2',
        winner_col: 'winner'
    }
    if venue_col:
        rename_dict[venue_col] = 'venue'
    if city_col:
        rename_dict[city_col] = 'city'
    
    df_selected = df_selected.rename(columns=rename_dict)
    
    print(f"\n✅ Selected {len(selected_columns)} columns for modeling\n")
    return df_selected


def handle_missing_values(df):
    """
    Handle missing values in the dataset
    
    Args:
        df (pd.DataFrame): Dataset with possible missing values
    
    Returns:
        pd.DataFrame: Dataset with missing values handled
    """
    print("🧹 STEP 2: Handling Missing Values")
    print("-" * 60)
    
    # Check for missing values
    missing_count = df.isnull().sum()
    total_missing = missing_count.sum()
    
    if total_missing == 0:
        print("✅ No missing values found!\n")
    else:
        print(f"⚠️  Found {total_missing} missing values:")
        print(missing_count[missing_count > 0])
    
    # Handle missing values
    # For categorical columns like team1, team2, winner - drop rows with missing values
    df_clean = df.dropna(subset=['team1', 'team2', 'winner'])
    
    # Remove rows where the winner is not a valid team
    if 'winner' in df_clean.columns:
        before_count = df_clean.shape[0]
        df_clean = df_clean[~df_clean['winner'].astype(str).str.strip().str.lower().isin(['unknown', ''])]
        dropped_unknown = before_count - df_clean.shape[0]
        if dropped_unknown > 0:
            print(f"\n⚠️  Removed {dropped_unknown} rows with invalid winner values (Unknown or blank)")
    
    # For venue/city - fill with 'Unknown' if missing
    if 'venue' in df_clean.columns:
        df_clean['venue'] = df_clean['venue'].fillna('Unknown')
    if 'city' in df_clean.columns:
        df_clean['city'] = df_clean['city'].fillna('Unknown')
    
    removed_rows = df.shape[0] - df_clean.shape[0]
    print(f"\n✅ Removed {removed_rows} rows with missing or invalid critical values")
    print(f"✅ Remaining rows: {df_clean.shape[0]}\n")
    
    return df_clean


def encode_categorical_data(df):
    """
    Convert team names (text) to numbers using LabelEncoder
    
    Machine learning models work with numbers, not text.
    This function converts:
    - 'CSK' -> 0, 'RCB' -> 1, 'MI' -> 2, etc.
    
    Args:
        df (pd.DataFrame): Dataset with text team names
    
    Returns:
        tuple: (encoded_df, encoders_dict) - processed data and encoding mappings
    """
    print("🔢 STEP 3: Encoding Categorical Data")
    print("-" * 60)
    print("Converting team names to numbers for ML model...\n")
    
    df_encoded = df.copy()
    encoders = {}
    
    # Encode team1
    le_team1 = LabelEncoder()
    df_encoded['team1_encoded'] = le_team1.fit_transform(df['team1'])
    encoders['team1'] = le_team1
    
    print(f"✅ Team 1 Encoding:")
    for i, team in enumerate(le_team1.classes_):
        print(f"   • {team} → {i}")
    
    # Encode team2
    le_team2 = LabelEncoder()
    df_encoded['team2_encoded'] = le_team2.fit_transform(df['team2'])
    encoders['team2'] = le_team2
    
    print(f"\n✅ Team 2 Encoding:")
    for i, team in enumerate(le_team2.classes_):
        print(f"   • {team} → {i}")
    
    # Encode winner (target variable)
    le_winner = LabelEncoder()
    df_encoded['winner_encoded'] = le_winner.fit_transform(df['winner'])
    encoders['winner'] = le_winner
    
    print(f"\n✅ Winner Encoding (Target):")
    for i, team in enumerate(le_winner.classes_):
        print(f"   • {team} → {i}")
    
    # Encode venue if it exists
    if 'venue' in df_encoded.columns:
        le_venue = LabelEncoder()
        df_encoded['venue_encoded'] = le_venue.fit_transform(df_encoded['venue'])
        encoders['venue'] = le_venue
        print(f"\n✅ Venue encoded ({len(le_venue.classes_)} venues)")
    
    # Encode city if it exists
    if 'city' in df_encoded.columns:
        le_city = LabelEncoder()
        df_encoded['city_encoded'] = le_city.fit_transform(df_encoded['city'])
        encoders['city'] = le_city
        print(f"✅ City encoded ({len(le_city.classes_)} cities)")
    
    print()
    return df_encoded, encoders


def prepare_features_target(df_encoded):
    """
    Prepare features (X) and target (y) for model training
    
    Args:
        df_encoded (pd.DataFrame): Encoded dataset
    
    Returns:
        tuple: (X, y) - features and target
    """
    print("📊 STEP 4: Preparing Features and Target")
    print("-" * 60)
    
    # Features (X) - what the model learns from
    feature_columns = ['team1_encoded', 'team2_encoded']
    if 'venue_encoded' in df_encoded.columns:
        feature_columns.append('venue_encoded')
    if 'city_encoded' in df_encoded.columns:
        feature_columns.append('city_encoded')
    
    X = df_encoded[feature_columns]
    
    # Target (y) - what we want to predict
    y = df_encoded['winner_encoded']
    
    print(f"✅ Features (X): {feature_columns}")
    print(f"✅ Target (y): winner_encoded")
    print(f"✅ Feature shape: {X.shape}")
    print(f"✅ Target shape: {y.shape}\n")
    
    return X, y, feature_columns


def preprocess_pipeline(df_raw):
    """
    Run complete preprocessing pipeline
    
    Args:
        df_raw (pd.DataFrame): Raw dataset
    
    Returns:
        tuple: (X, y, encoders, feature_columns) - ready for model training
    """
    print("\n" + "=" * 70)
    print("📋 DATA PREPROCESSING PIPELINE")
    print("=" * 70 + "\n")
    
    # Step 0: Build match summary if dataset is ball-by-ball
    df_prepared = build_match_summary(df_raw)
    
    # Step 1: Select columns
    df_selected = select_important_columns(df_prepared)
    if df_selected is None:
        return None, None, None, None
    
    # Step 2: Handle missing values
    df_clean = handle_missing_values(df_selected)
    
    # Step 3: Encode categories
    df_encoded, encoders = encode_categorical_data(df_clean)
    
    # Step 4: Prepare features and target
    X, y, feature_columns = prepare_features_target(df_encoded)
    
    print("=" * 70)
    print("✅ PREPROCESSING COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Ready for model training:")
    print(f"   • Features shape: {X.shape}")
    print(f"   • Target shape: {y.shape}")
    print(f"   • Feature columns: {feature_columns}\n")
    
    return X, y, encoders, feature_columns


def main():
    """Main function to run preprocessing"""
    # Load data
    df = load_data()
    if df is None:
        return None, None, None, None
    
    # Run preprocessing
    X, y, encoders, feature_columns = preprocess_pipeline(df)
    
    return X, y, encoders, feature_columns


if __name__ == "__main__":
    main()
