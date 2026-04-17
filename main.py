"""
Main Entry Point for IPL Prediction Project
Run this file to execute the complete pipeline
"""

import sys
import os

# Add src directory to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from data_loader import load_data, explore_data
from preprocessing import preprocess_pipeline
from model_trainer import train_pipeline


def main():
    print("\n" + "=" * 70)
    print("🏏 IPL 2026 MATCH WINNER PREDICTION")
    print("=" * 70)
    
    # STEP 2: Load and explore data
    print("\n📍 STEP 2: Data Loading and Exploration\n")
    df = load_data()
    
    if df is None:
        print("\n❌ Failed to load data. Please check your CSV file.\n")
        return None, None, None, None
    
    explore_data(df)
    
    # STEP 3: Preprocess data
    print("\n📍 STEP 3: Data Preprocessing\n")
    X, y, encoders, feature_columns = preprocess_pipeline(df)
    
    if X is None:
        print("\n❌ Failed to preprocess data.\n")
        return None, None, None, None
    
    # STEP 4: Train model
    print("\n📍 STEP 4: Model Training\n")
    model, metrics = train_pipeline(X, y, encoders, feature_columns)
    
    if model is None:
        print("\n❌ Failed to train model.\n")
        return None, None, None, None
    
    return model, encoders, feature_columns, metrics

if __name__ == "__main__":
    model, encoders, feature_columns, metrics = main()
    
    if model is not None:
        print("\n✅ Complete pipeline executed successfully!")
        print(f"   • Test Accuracy: {metrics['test_accuracy']:.2%}")
        print(f"   • Model saved and ready for predictions!\n")


