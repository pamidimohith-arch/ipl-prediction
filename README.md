# IPL 2026 Match Winner Prediction

A beginner-friendly Python project that uses historical IPL data to predict match winners using machine learning.

## Project Overview

This project uses an IPL dataset and a Random Forest classifier to predict the likely winner of a match based on:
- batting team
- bowling team
- venue
- city

The project includes:
- data loading and exploration
- preprocessing of match-level data
- model training and evaluation
- saved model files for later prediction
- a prediction script for interactive use

## Folder Structure

```
IPL PREDICTION 2026/
├── data/                  # IPL dataset file
├── models/                # Saved trained model and encoders
├── notebooks/             # Optional exploration notebooks
├── src/                   # Python source code modules
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model_trainer.py
│   └── predictor.py
├── main.py                # Main training pipeline
├── predict.py             # Interactive prediction script
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── .gitignore             # Files to ignore in git
├── SETUP_GUIDE.md         # Setup instructions
└── STEP_5_EVALUATION_GUIDE.md # Evaluation explanation
```

## Setup Instructions

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```

2. Activate the environment:
   ```bash
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Use

### Train the model

Run the full training and preprocessing pipeline:

```bash
python3 main.py
```

### Predict a match winner

Use the interactive script after training has completed:

```bash
python3 predict.py
```

Follow the prompts to enter team names, venue, and city.

## Important Files

- `src/data_loader.py`: loads and explores the CSV dataset
- `src/preprocessing.py`: cleans and encodes match data
- `src/model_trainer.py`: trains and evaluates the Random Forest model
- `src/predictor.py`: loads the saved model and makes predictions
- `main.py`: runs the full training pipeline
- `predict.py`: interactive prediction script

## Uploading to GitHub

1. Create a new GitHub repository on GitHub.com.
2. Copy the repository URL.
3. Run these commands in the project folder:

```bash
git init
git add .
git commit -m "Initial project commit"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

Replace `<YOUR_GITHUB_REPO_URL>` with the repository address from GitHub.

## Notes

- Keep the `data/` folder private if you do not want to upload the dataset file.
- The model can be improved with extra match-level features, toss data, and team form history.

## Future Improvements

- Add a Streamlit web app for easy user interaction
- Improve the model using more match and team features
- Add a better prediction UI and GitHub portfolio screenshots
# ipl-prediction
