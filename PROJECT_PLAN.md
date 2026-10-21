# FootyPredict — Project Plan & Technical Blueprint
### Dual-Stage Machine Learning Framework for Football Match Outcome and Scoreline Prediction

**Student:** Saad Vakharia &nbsp;|&nbsp; **Roll No:** 155 &nbsp;|&nbsp; **Course:** Artificial Intelligence & Data Science II (AI & DS II)  
**Institution:** Atharva College of Engineering, University of Mumbai &nbsp;|&nbsp; **Academic Year:** 2026–2027  
**Department:** Information Technology &nbsp;|&nbsp; **Guide:** Prof. Suvarna Panchal  

---

## 1. System Architecture & Objectives

FootyPredict decouples match forecasting into two mathematically distinct stages rather than forcing a single model to learn disparate objectives:

```
[ Match Context & History ]
       │
       ├── Rolling 5-Match Form (Points, Goal Diff, xG)
       ├── Dynamic Elo Rating Differential
       ├── Head-to-Head (H2H) Historical Bias
       └── Home Advantage & Rest Day Differential
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DUAL-STAGE INFERENCE ENGINE                  │
├────────────────────────────────┬────────────────────────────────┤
│            STAGE 1             │            STAGE 2             │
│   Discrete Outcome Classifier  │    Bivariate Poisson Model     │
│   (Home Win / Draw / Away Win) │    (Expected Goals & Score)    │
│                                │                                │
│ • Evaluated Models:            │ • Dynamic Attack / Defense     │
│   - Logistic Regression        │   intensity parameters         │
│   - Random Forest Classifier   │ • Joint probability matrix:    │
│   - Gradient Boosted Trees     │   P(X=x, Y=y) = P(x;λ_h)P(y;λ_a)│
│ • Key Metrics:                 │ • Outputs:                     │
│   Accuracy, Precision, Recall, │   Team xG, Exact Scoreline     │
│   F1-Score (Macro & Weighted)  │   Probability Distribution     │
└────────────────────────────────┴────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    HIGH-SIGNAL WEB DASHBOARD                    │
│   Real-time probabilities, Scoreline heatmaps, Validation radar │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Technical Stack

| Layer | Technologies |
| :--- | :--- |
| **Data Processing & Analytics** | Python 3.14, NumPy, Pandas, SciPy |
| **Machine Learning** | Scikit-Learn (LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, HistGradientBoosting) |
| **Statistical Modeling** | Poisson Distribution, Bivariate Scoreline Matrix Formulation |
| **Evaluation Suite** | Accuracy, Macro F1-Score, Weighted F1-Score, Precision, Recall, Confusion Matrix, ROC-AUC |
| **Web Interface** | Modern Vanilla HTML5 / ES6 JavaScript / Precision CSS (Zero AI-slop, matte obsidian palette, sharp borders) |
| **Documentation & Reporting** | 5-Page Faculty Standard Word Report (`docx`) with IEEE citations & Certificate |

---

## 3. Directory Layout

```
AI_DS_II/
├── data/
│   ├── raw/                      # Raw league match fixtures
│   └── matches.csv               # Processed multi-season match records with engineered features
├── models/
│   ├── outcome_model.pkl         # Trained Stage-1 Classifier
│   └── scoreline_model.pkl       # Trained Stage-2 Poisson Regressor
├── notebooks/
│   └── footypredict_pipeline.ipynb # End-to-end EDA, model benchmarking, and metrics
├── src/
│   ├── data_loader.py            # Feature engineering (Rolling form, Elo ratings, H2H)
│   ├── model.py                  # Dual-stage classifier & Poisson scoreline engine
│   ├── train.py                  # Multi-model training and hyperparameter evaluation
│   └── evaluate.py               # Generates confusion matrices, ROC curves, F1-score tables
├── app/
│   ├── index.html                # High-signal sports analytics dashboard
│   ├── app.css                   # Obsidian/slate theme, 4px crisp radius, hairline borders
│   └── app.js                    # Match simulation engine and dynamic probability charts
└── report/
    ├── figures/                  # High-resolution evaluation charts
    └── FootyPredict_Mini_Project_Report.docx # Formatted 5-page submission document
```

---

## 4. Key Metrics & Benchmarks

Football match prediction operates under high inherent variance (inherent match entropy ~60-65% theoretical predictability). The models are evaluated on:

- **Accuracy:** Percentage of correct match outcome predictions across test split.
- **Precision (Macro & Weighted):** Exactness per class (Home Win, Draw, Away Win) to avoid draw-suppression bias.
- **Recall (Macro & Weighted):** Completeness of predictions per outcome class.
- **F1-Score (Primary Faculty Criterion):** Harmonic mean of Precision and Recall.
- **Log-Loss / Brier Score:** Calibration of probability estimates for fair confidence assessments.
