"""
Interactive Prediction Script
Run this script to predict IPL match winners using the saved model.
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from predictor import load_saved_model, print_available_teams, predict_match_winner


def main():
    print("\n" + "=" * 70)
    print("🏏 IPL MATCH WINNER PREDICTION - PREDICTION MODE")
    print("=" * 70)

    model, encoders, feature_columns = load_saved_model()
    if model is None:
        return

    print_available_teams(encoders)

    team1 = input("Enter Team 1 (batting team): ").strip()
    team2 = input("Enter Team 2 (bowling team): ").strip()
    venue = input("Enter Venue (optional, press Enter to skip): ").strip()
    city = input("Enter City (optional, press Enter to skip): ").strip()

    if venue == "":
        venue = None
    if city == "":
        city = None

    predicted_winner = predict_match_winner(model, encoders, feature_columns, team1, team2, venue=venue, city=city)
    if predicted_winner is not None:
        print("\n" + "-" * 70)
        print(f"✅ Predicted Winner: {predicted_winner}")
        print("-" * 70)


if __name__ == "__main__":
    main()
