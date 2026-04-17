"""
Model Training Module
This module trains a Random Forest classifier to predict IPL match winners
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os
import sys

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_PATH = os.path.join(PROJECT_ROOT, 'models', 'ipl_model.pkl')
ENCODERS_PATH = os.path.join(PROJECT_ROOT, 'models', 'encoders.pkl')


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    
    Why split?
    - Training set: Model learns from this data
    - Testing set: Model's performance is evaluated on new, unseen data
    - test_size=0.2 means 80% train, 20% test (common practice)
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        test_size (float): Proportion of test data (0.2 = 20%)
        random_state (int): For reproducibility
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("🔀 STEP 1: Splitting Data into Train and Test Sets")
    print("-" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Ensures both sets have similar class distribution
    )
    
    print(f"✅ Data Split Complete:")
    print(f"   • Training set: {X_train.shape[0]} matches (80%)")
    print(f"   • Testing set: {X_test.shape[0]} matches (20%)")
    print(f"   • Features per match: {X_train.shape[1]}")
    print(f"   • Train-Test Ratio: {X_train.shape[0]}:{X_test.shape[0]}\n")
    
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest Classifier
    
    What is Random Forest?
    - Ensemble method: Creates multiple decision trees
    - Each tree votes on the winner
    - Final prediction: Majority vote wins
    - Very effective for classification tasks!
    
    Parameters:
    - n_estimators=100: Number of trees (more trees = more accuracy, slower)
    - max_depth=10: How deep each tree can go (prevents overfitting)
    - min_samples_split=5: Min samples to split a node (prevents overfitting)
    - random_state=42: For reproducibility
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
    
    Returns:
        RandomForestClassifier: Trained model
    """
    print("🤖 STEP 2: Training Random Forest Model")
    print("-" * 60)
    
    print("Creating Random Forest with 100 decision trees...\n")
    
    # Create the model
    model = RandomForestClassifier(
        n_estimators=100,        # 100 trees
        max_depth=10,            # Max depth of each tree
        min_samples_split=5,     # Min samples to split
        min_samples_leaf=2,      # Min samples in leaf node
        random_state=42,         # For reproducibility
        n_jobs=-1                # Use all CPU cores
    )
    
    # Train the model
    print("Training model on historical data...")
    model.fit(X_train, y_train)
    
    print("✅ Model Training Complete!\n")
    
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate model performance on both training and testing data
    
    Metrics Explained:
    - Accuracy: % of correct predictions
    - Precision: Of predicted winners, how many were correct?
    - Recall: Of actual winners, how many did we identify?
    - F1-Score: Balance between precision and recall
    
    Args:
        model: Trained RandomForestClassifier
        X_train, X_test: Features
        y_train, y_test: Targets
    
    Returns:
        dict: Dictionary with all metrics
    """
    print("📊 STEP 3: Model Evaluation")
    print("-" * 60)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    
    print(f"📈 Training Accuracy: {train_accuracy:.2%}")
    print(f"   (How well model learned training data)")
    
    print(f"\n📉 Testing Accuracy: {test_accuracy:.2%}")
    print(f"   (How well model predicts NEW, unseen data)")
    
    print(f"\n🎯 Precision: {test_precision:.2%}")
    print(f"   (When model predicts a winner, is it correct?)")
    
    print(f"\n📍 Recall: {test_recall:.2%}")
    print(f"   (Does model find all the winners?)")
    
    print(f"\n⚖️  F1-Score: {test_f1:.2%}")
    print(f"   (Overall model quality)\n")
    
    # Interpretation
    print("🔍 Model Interpretation:")
    if test_accuracy > 0.85:
        print("   ✅ EXCELLENT! Model is highly accurate!")
    elif test_accuracy > 0.75:
        print("   ✅ GOOD! Model performs well!")
    elif test_accuracy > 0.65:
        print("   ⚠️  OKAY! Model works but could improve.")
    else:
        print("   ❌ NEEDS IMPROVEMENT! Consider more data or features.")
    
    if train_accuracy - test_accuracy > 0.15:
        print("   ⚠️  WARNING: Overfitting detected! Model memorized training data.")
    
    print()
    
    metrics = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': test_f1,
        'confusion_matrix': conf_matrix
    }
    
    return metrics


def feature_importance(model, feature_names):
    """
    Show which features are most important for predictions
    
    Args:
        model: Trained RandomForestClassifier
        feature_names (list): Names of features
    """
    print("🎯 STEP 4: Feature Importance Analysis")
    print("-" * 60)
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("Features ranked by importance for prediction:\n")
    
    for i, idx in enumerate(indices):
        print(f"   {i+1}. {feature_names[idx]}: {importances[idx]:.2%}")
    
    print()


def save_model(model, encoders, feature_columns):
    """
    Save trained model and encoders to disk
    
    Why save?
    - Don't retrain every time the program runs
    - Can use the model later for predictions
    - Can share model with others
    
    Args:
        model: Trained RandomForestClassifier
        encoders: Dictionary of LabelEncoders
        feature_columns: List of feature column names
    """
    print("💾 STEP 5: Saving Model")
    print("-" * 60)
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(MODELS_PATH), exist_ok=True)
    
    # Save model
    joblib.dump(model, MODELS_PATH)
    print(f"✅ Model saved to: {MODELS_PATH}")
    
    # Save encoders
    model_data = {
        'encoders': encoders,
        'feature_columns': feature_columns
    }
    joblib.dump(model_data, ENCODERS_PATH)
    print(f"✅ Encoders saved to: {ENCODERS_PATH}")
    
    print(f"✅ Model ready for predictions!\n")


def load_model():
    """
    Load previously trained model and encoders
    
    Returns:
        tuple: (model, encoders, feature_columns) or (None, None, None) if not found
    """
    if not os.path.exists(MODELS_PATH) or not os.path.exists(ENCODERS_PATH):
        return None, None, None
    
    model = joblib.load(MODELS_PATH)
    model_data = joblib.load(ENCODERS_PATH)
    
    return model, model_data['encoders'], model_data['feature_columns']


def train_pipeline(X, y, encoders, feature_columns):
    """
    Run complete model training pipeline
    
    Args:
        X: Features
        y: Target
        encoders: Dictionary of LabelEncoders
        feature_columns: List of feature names
    
    Returns:
        tuple: (model, metrics)
    """
    print("\n" + "=" * 70)
    print("🤖 MODEL TRAINING PIPELINE")
    print("=" * 70 + "\n")
    
    # Step 1: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 2: Train model
    model = train_random_forest(X_train, y_train)
    
    # Step 3: Evaluate model
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Step 4: Show feature importance
    feature_importance(model, feature_columns)
    
    # Step 5: Save model
    save_model(model, encoders, feature_columns)
    
    print("=" * 70)
    print("✅ MODEL TRAINING COMPLETE!")
    print("=" * 70 + "\n")
    
    return model, metrics


def main():
    """Main function to train model"""
    from preprocessing import preprocess_pipeline
    from data_loader import load_data
    
    # Load and preprocess data
    df = load_data()
    if df is None:
        return None, None
    
    X, y, encoders, feature_columns = preprocess_pipeline(df)
    if X is None:
        return None, None
    
    # Train model
    model, metrics = train_pipeline(X, y, encoders, feature_columns)
    
    return model, metrics


if __name__ == "__main__":
    main()
