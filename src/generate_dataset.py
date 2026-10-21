"""
FootyPredict: Realistic Multi-Season European Football Dataset Generator
Generates realistic match statistics, rolling momentum, Elo differentials,
and dual-stage outcomes (FTR and Full Time Scorelines).
"""

import os
import csv
import math
import random
from datetime import datetime, timedelta

TEAMS_DATA = {
    "Premier League": [
        {"name": "Arsenal", "base_elo": 1890, "att": 2.15, "defn": 0.75},
        {"name": "Manchester City", "base_elo": 1920, "att": 2.30, "defn": 0.80},
        {"name": "Liverpool", "base_elo": 1905, "att": 2.25, "defn": 0.82},
        {"name": "Chelsea", "base_elo": 1780, "att": 1.75, "defn": 1.10},
        {"name": "Tottenham", "base_elo": 1765, "att": 1.80, "defn": 1.25},
        {"name": "Aston Villa", "base_elo": 1770, "att": 1.70, "defn": 1.15},
        {"name": "Newcastle United", "base_elo": 1750, "att": 1.65, "defn": 1.18},
        {"name": "Manchester United", "base_elo": 1740, "att": 1.55, "defn": 1.22},
        {"name": "Brighton", "base_elo": 1710, "att": 1.50, "defn": 1.30},
        {"name": "West Ham", "base_elo": 1680, "att": 1.35, "defn": 1.40},
    ],
    "La Liga": [
        {"name": "Real Madrid", "base_elo": 1930, "att": 2.35, "defn": 0.78},
        {"name": "Barcelona", "base_elo": 1910, "att": 2.40, "defn": 0.85},
        {"name": "Atletico Madrid", "base_elo": 1820, "att": 1.75, "defn": 0.80},
        {"name": "Athletic Bilbao", "base_elo": 1740, "att": 1.55, "defn": 1.05},
        {"name": "Real Sociedad", "base_elo": 1725, "att": 1.45, "defn": 1.10},
        {"name": "Villarreal", "base_elo": 1715, "att": 1.50, "defn": 1.30},
    ],
    "Bundesliga": [
        {"name": "Bayern Munich", "base_elo": 1915, "att": 2.50, "defn": 0.90},
        {"name": "Bayer Leverkusen", "base_elo": 1860, "att": 2.10, "defn": 0.92},
        {"name": "Borussia Dortmund", "base_elo": 1800, "att": 1.95, "defn": 1.20},
        {"name": "RB Leipzig", "base_elo": 1785, "att": 1.85, "defn": 1.15},
    ],
    "Serie A": [
        {"name": "Inter Milan", "base_elo": 1875, "att": 2.05, "defn": 0.76},
        {"name": "Juventus", "base_elo": 1810, "att": 1.65, "defn": 0.82},
        {"name": "AC Milan", "base_elo": 1800, "att": 1.80, "defn": 1.15},
        {"name": "Napoli", "base_elo": 1795, "att": 1.75, "defn": 1.05},
        {"name": "Atalanta", "base_elo": 1780, "att": 1.90, "defn": 1.25},
    ]
}

def sample_poisson(lam):
    """Knuth's algorithm for generating Poisson distributed random variables."""
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= random.random()
    return k - 1

def generate_matches(num_matches=1800, seed=42):
    random.seed(seed)
    all_teams = []
    for league, teams in TEAMS_DATA.items():
        for t in teams:
            t_copy = t.copy()
            t_copy["league"] = league
            all_teams.append(t_copy)

    # Initialize dynamic state
    team_state = {}
    for t in all_teams:
        team_state[t["name"]] = {
            "elo": t["base_elo"],
            "recent_pts": [random.choice([0, 1, 3]) for _ in range(5)],
            "recent_gd": [random.randint(-2, 3) for _ in range(5)],
            "last_match_date": datetime(2023, 8, 1) + timedelta(days=random.randint(1, 10))
        }

    records = []
    start_date = datetime(2023, 8, 12)

    for match_id in range(1, num_matches + 1):
        # Pick league and two distinct teams
        league = random.choices(
            ["Premier League", "La Liga", "Serie A", "Bundesliga"],
            weights=[0.45, 0.25, 0.20, 0.10]
        )[0]
        league_teams = [t for t in all_teams if t["league"] == league]
        home_t, away_t = random.sample(league_teams, 2)
        h_name, a_name = home_t["name"], away_t["name"]

        h_st = team_state[h_name]
        a_st = team_state[a_name]

        # Match date
        match_date = start_date + timedelta(days=int((match_id / num_matches) * 700))
        rest_h = min(14, max(3, (match_date - h_st["last_match_date"]).days))
        rest_a = min(14, max(3, (match_date - a_st["last_match_date"]).days))
        h_st["last_match_date"] = match_date
        a_st["last_match_date"] = match_date

        # Elo & Features
        h_elo = round(h_st["elo"], 1)
        a_elo = round(a_st["elo"], 1)
        elo_diff = round(h_elo - a_elo, 1)

        h_form_pts = sum(h_st["recent_pts"][-5:])
        a_form_pts = sum(a_st["recent_pts"][-5:])
        h_form_gd = sum(h_st["recent_gd"][-5:])
        a_form_gd = sum(a_st["recent_gd"][-5:])

        # Calculate Expected Goals (xG) using team attacking strength, defense weakness, and home advantage
        home_adv = 0.28  # Home pitch goal advantage
        rest_factor_h = 0.05 if rest_h >= 6 else -0.05
        rest_factor_a = 0.05 if rest_a >= 6 else -0.05

        lam_home = max(0.4, (home_t["att"] * 0.7 + away_t["defn"] * 0.45) + home_adv + rest_factor_h + (elo_diff / 500.0) * 0.35)
        lam_away = max(0.3, (away_t["att"] * 0.65 + home_t["defn"] * 0.40) + rest_factor_a - (elo_diff / 500.0) * 0.30)

        # Introduce slight correlation / variance
        fthg = sample_poisson(lam_home)
        ftag = sample_poisson(lam_away)

        if fthg > ftag:
            ftr = "H"
            h_pts, a_pts = 3, 0
        elif fthg < ftag:
            ftr = "A"
            h_pts, a_pts = 0, 3
        else:
            ftr = "D"
            h_pts, a_pts = 1, 1

        # Update dynamic state
        h_st["recent_pts"].append(h_pts)
        a_st["recent_pts"].append(a_pts)
        h_st["recent_gd"].append(fthg - ftag)
        a_st["recent_gd"].append(ftag - fthg)

        # Elo rating update (K-factor = 24)
        expected_h = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))
        actual_h = 1.0 if ftr == "H" else (0.5 if ftr == "D" else 0.0)
        h_st["elo"] += 24 * (actual_h - expected_h)
        a_st["elo"] -= 24 * (actual_h - expected_h)

        # Additional match statistics for depth
        home_shots = max(fthg + 1, int(lam_home * 4.2 + random.randint(1, 8)))
        away_shots = max(ftag + 1, int(lam_away * 3.8 + random.randint(1, 7)))
        home_sot = min(home_shots, max(fthg, int(home_shots * random.uniform(0.32, 0.55))))
        away_sot = min(away_shots, max(ftag, int(away_shots * random.uniform(0.30, 0.52))))
        home_corners = random.randint(2, 11)
        away_corners = random.randint(1, 9)

        records.append({
            "MatchID": f"M{match_id:04d}",
            "Date": match_date.strftime("%Y-%m-%d"),
            "League": league,
            "HomeTeam": h_name,
            "AwayTeam": a_name,
            "HomeElo": h_elo,
            "AwayElo": a_elo,
            "EloDiff": elo_diff,
            "HomeFormPts5": h_form_pts,
            "AwayFormPts5": a_form_pts,
            "HomeFormGD5": h_form_gd,
            "AwayFormGD5": a_form_gd,
            "RestDaysHome": rest_h,
            "RestDaysAway": rest_a,
            "HomeShots": home_shots,
            "AwayShots": away_shots,
            "HomeShotsOnTarget": home_sot,
            "AwayShotsOnTarget": away_sot,
            "HomeCorners": home_corners,
            "AwayCorners": away_corners,
            "HomexG": round(lam_home, 2),
            "AwayxG": round(lam_away, 2),
            "FTHG": fthg,
            "FTAG": ftag,
            "FTR": ftr
        })

    return records

def save_csv(records, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fieldnames = list(records[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"[OK] Generated {len(records)} matches -> {filepath}")

if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "matches.csv")
    recs = generate_matches(num_matches=2200, seed=155)
    save_csv(recs, out_path)
