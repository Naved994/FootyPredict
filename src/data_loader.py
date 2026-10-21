"""
FootyPredict: Data Loader & Feature Engineering Pipeline
Handles loading match records, feature extraction, scaling, and train-test splitting.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

FEATURE_COLS = [
    "EloDiff",
    "HomeElo",
    "AwayElo",
    "HomeFormPts5",
    "AwayFormPts5",
    "HomeFormGD5",
    "AwayFormGD5",
    "RestDaysHome",
    "RestDaysAway",
    "HomeShots",
    "AwayShots",
    "HomeShotsOnTarget",
    "AwayShotsOnTarget",
    "HomeCorners",
    "AwayCorners"
]

OUTCOME_MAPPING = {"H": 0, "D": 1, "A": 2}
INV_OUTCOME_MAPPING = {0: "Home Win", 1: "Draw", 2: "Away Win"}

def load_data(filepath=None):
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), "..", "data", "matches.csv")
    df = pd.read_csv(filepath)
    return df

def prepare_features(df):
    """
    Extracts numerical feature matrix X, outcome labels y_outcome,
    and goal targets y_home_goals, y_away_goals.
    """
    X = df[FEATURE_COLS].copy()
    y_outcome = df["FTR"].map(OUTCOME_MAPPING).values
    y_home_goals = df["FTHG"].values
    y_away_goals = df["FTAG"].values
    return X, y_outcome, y_home_goals, y_away_goals

def split_and_scale(X, y_outcome, y_home_goals, y_away_goals, test_size=0.2, random_state=42):
    """
    Stratified split on outcome, with feature standardization.
    """
    X_train, X_test, y_out_train, y_out_test, y_hg_train, y_hg_test, y_ag_train, y_ag_test = train_test_split(
        X, y_outcome, y_home_goals, y_away_goals,
        test_size=test_size,
        random_state=random_state,
        stratify=y_outcome
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "y_out_train": y_out_train,
        "y_out_test": y_out_test,
        "y_hg_train": y_hg_train,
        "y_hg_test": y_hg_test,
        "y_ag_train": y_ag_train,
        "y_ag_test": y_ag_test,
        "scaler": scaler,
        "feature_names": FEATURE_COLS
    }

if __name__ == "__main__":
    df = load_data()
    print(f"Loaded dataset: {df.shape[0]} matches, {df.shape[1]} columns.")
    X, y_out, y_hg, y_ag = prepare_features(df)
    data = split_and_scale(X, y_out, y_hg, y_ag)
    print(f"Train samples: {data['X_train'].shape[0]}, Test samples: {data['X_test'].shape[0]}")
