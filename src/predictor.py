"""
Prediction Module
Loads a saved IPL winner model and predicts the winner for a pair of teams.
"""

import os
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Models directory paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_PATH = os.path.join(PROJECT_ROOT, 'models', 'ipl_model.pkl')
ENCODERS_PATH = os.path.join(PROJECT_ROOT, 'models', 'encoders.pkl')


def load_saved_model():
    """Load the saved model and encoders from disk."""
    if not os.path.exists(MODELS_PATH) or not os.path.exists(ENCODERS_PATH):
        print("❌ Saved model files not found.")
        print("Please run `python main.py` first to train and save the model.")
        return None, None, None

    model = joblib.load(MODELS_PATH)
    model_data = joblib.load(ENCODERS_PATH)
    encoders = model_data.get('encoders')
    feature_columns = model_data.get('feature_columns')
    return model, encoders, feature_columns


def normalize_team_name(team_name):
    """Normalize team name text to compare user input robustly."""
    if not isinstance(team_name, str):
        return team_name
    return team_name.strip().lower().replace(' ', ' ')


def get_available_teams(encoders):
    """Return the sorted list of all teams available in the encoder."""
    if encoders is None or 'team1' not in encoders:
        return []
    return list(encoders['team1'].classes_)


def find_team_name(user_input, encoder):
    """Find the closest matching team name from the encoder classes."""
    if user_input is None or encoder is None:
        return None
    user_input_clean = user_input.strip().lower()
    for team in encoder.classes_:
        if team.strip().lower() == user_input_clean:
            return team
    # allow partial matching if exact match fails
    for team in encoder.classes_:
        lower_team = team.strip().lower()
        if user_input_clean in lower_team or lower_team in user_input_clean:
            return team
    return None


def predict_match_winner(model, encoders, feature_columns, team1, team2, venue=None, city=None):
    """Predict the winner for a given match scenario."""
    if model is None or encoders is None or feature_columns is None:
        print("❌ Model or encoders are not loaded.")
        return None

    # Validate teams with encoder
    team1_name = find_team_name(team1, encoders['team1'])
    team2_name = find_team_name(team2, encoders['team2'])
    if team1_name is None:
        print(f"❌ Team 1 '{team1}' is not in the trained dataset.")
        return None
    if team2_name is None:
        print(f"❌ Team 2 '{team2}' is not in the trained dataset.")
        return None
    if team1_name == team2_name:
        print("❌ Team 1 and Team 2 must be different.")
        return None

    # Build feature row
    input_row = {}
    input_row['team1_encoded'] = encoders['team1'].transform([team1_name])[0]
    input_row['team2_encoded'] = encoders['team2'].transform([team2_name])[0]

    if 'venue_encoded' in feature_columns:
        venue_value = venue if venue is not None else 'Unknown'
        venue_name = venue_value.strip()
        if venue_name == '':
            venue_name = 'Unknown'
        if 'venue' in encoders and venue_name in encoders['venue'].classes_:
            input_row['venue_encoded'] = encoders['venue'].transform([venue_name])[0]
        else:
            print(f"⚠️  Venue '{venue_name}' is not available in the saved model. Using 'Unknown'.")
            if 'Unknown' in encoders['venue'].classes_:
                input_row['venue_encoded'] = encoders['venue'].transform(['Unknown'])[0]
            else:
                input_row['venue_encoded'] = 0

    if 'city_encoded' in feature_columns:
        city_value = city if city is not None else 'Unknown'
        city_name = city_value.strip()
        if city_name == '':
            city_name = 'Unknown'
        if 'city' in encoders and city_name in encoders['city'].classes_:
            input_row['city_encoded'] = encoders['city'].transform([city_name])[0]
        else:
            print(f"⚠️  City '{city_name}' is not available in the saved model. Using 'Unknown'.")
            if 'Unknown' in encoders['city'].classes_:
                input_row['city_encoded'] = encoders['city'].transform(['Unknown'])[0]
            else:
                input_row['city_encoded'] = 0

    # Make sure columns are in correct order
    X_input = np.array([input_row[col] for col in feature_columns]).reshape(1, -1)
    winner_encoded = model.predict(X_input)[0]
    predicted_winner = encoders['winner'].inverse_transform([winner_encoded])[0]
    return predicted_winner


def print_available_teams(encoders):
    teams = get_available_teams(encoders)
    if not teams:
        print("No team list available.")
        return
    print("\nAvailable teams:")
    for team in teams:
        print(f"- {team}")
    print()
