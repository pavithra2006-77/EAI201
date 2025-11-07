# fifa_prediction_app.py

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, auc, classification_report)

# ------------------ CONFIG ------------------
BASE_DIR = r"C:\Users\test\OneDrive\Desktop\assignment2aiml"
CLEANED_FILE = "cleaned_fifa_dataset.csv"
PATH_CLEANED = os.path.join(BASE_DIR, CLEANED_FILE)

FEATURES = ['Rank Difference', 'Points Difference',
            'Home Team Avg Age', 'Away Team Avg Age',
            'Home Team Experience', 'Away Team Experience',
            'Home Team Win Rate', 'Away Team Win Rate']
TARGET = 'Home_Win'

# ------------------ SCRAPER HANDLER ------------------
def run_scraper_cli():
    try:
        from web_scraper import run_scraper
        run_scraper()
        print(" Scraper executed successfully.")
    except ModuleNotFoundError as e:
        print(f" Scraper skipped: {e}")
    except Exception as e:
        print(f" Scraper encountered an error: {e}")

# ------------------ LOAD DATASET ------------------
def load_dataset():
    if not os.path.exists(PATH_CLEANED):
        print(f" Cleaned dataset not found at {PATH_CLEANED}")
        return None
    df = pd.read_csv(PATH_CLEANED)
    print(f" Cleaned dataset loaded successfully ({df.shape[0]} rows, {df.shape[1]} columns).")
    print("\nSample data:")
    print(df.head())
    return df

# ------------------ TRAIN MODELS ------------------
def train_models(df):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Logistic Regression ---
    log_params = {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs']}
    log_grid = GridSearchCV(LogisticRegression(max_iter=5000), log_params, cv=5, scoring='accuracy')
    log_grid.fit(X_train_scaled, y_train)
    best_log_model = log_grid.best_estimator_
    print("\n Logistic Regression trained. Best Params:", log_grid.best_params_)

    # --- Random Forest ---
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='accuracy')
    rf_grid.fit(X_train, y_train)
    best_rf_model = rf_grid.best_estimator_
    print("\n Random Forest trained. Best Params:", rf_grid.best_params_)

    return best_log_model, best_rf_model, scaler, X_test, y_test, X, y

# ------------------ EVALUATE MODELS ------------------
def evaluate_models(best_log_model, best_rf_model, scaler, X_test, y_test, X, y):
    # --- Logistic Regression ---
    X_test_scaled = scaler.transform(X_test)
    y_pred_log = best_log_model.predict(X_test_scaled)
    y_prob_log = best_log_model.predict_proba(X_test_scaled)

    acc_log = accuracy_score(y_test, y_pred_log)
    prec_log = precision_score(y_test, y_pred_log, average='weighted')
    rec_log = recall_score(y_test, y_pred_log, average='weighted')
    f1_log = f1_score(y_test, y_pred_log, average='weighted')
    roc_log = roc_auc_score(y_test, y_prob_log, multi_class='ovr')

    # --- Random Forest ---
    y_pred_rf = best_rf_model.predict(X_test)
    y_prob_rf = best_rf_model.predict_proba(X_test)

    acc_rf = accuracy_score(y_test, y_pred_rf)
    prec_rf = precision_score(y_test, y_pred_rf, average='weighted')
    rec_rf = recall_score(y_test, y_pred_rf, average='weighted')
    f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
    roc_rf = roc_auc_score(y_test, y_prob_rf, multi_class='ovr')

    print("\n=== MODEL PERFORMANCE METRICS ===")
    print(f"{'Metric':<15} {'Logistic Regression':<20} {'Random Forest':<20}")
    print("-" * 60)
    print(f"{'Accuracy':<15} {acc_log:.4f} {'':5} {acc_rf:.4f}")
    print(f"{'Precision':<15} {prec_log:.4f} {'':5} {prec_rf:.4f}")
    print(f"{'Recall':<15} {rec_log:.4f} {'':5} {rec_rf:.4f}")
    print(f"{'F1-Score':<15} {f1_log:.4f} {'':5} {f1_rf:.4f}")
    print(f"{'ROC-AUC':<15} {roc_log:.4f} {'':5} {roc_rf:.4f}")

    # --- Confusion Matrices ---
    cm_log = confusion_matrix(y_test, y_pred_log)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues')
    plt.title('Logistic Regression Confusion Matrix')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens')
    plt.title('Random Forest Confusion Matrix')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout(); plt.show()

    # --- ROC Curves ---
    classes = sorted(y.unique())
    y_test_bin = label_binarize(y_test, classes=classes)

    fpr_log, tpr_log, roc_auc_log = dict(), dict(), dict()
    fpr_rf, tpr_rf, roc_auc_rf = dict(), dict(), dict()

    for i, cls in enumerate(classes):
        fpr_log[i], tpr_log[i], _ = roc_curve(y_test_bin[:, i], y_prob_log[:, i])
        roc_auc_log[i] = auc(fpr_log[i], tpr_log[i])
        fpr_rf[i], tpr_rf[i], _ = roc_curve(y_test_bin[:, i], y_prob_rf[:, i])
        roc_auc_rf[i] = auc(fpr_rf[i], tpr_rf[i])

    plt.figure(figsize=(10, 6))
    for i, cls in enumerate(classes):
        plt.plot(fpr_log[i], tpr_log[i], label=f'LogReg (Class {cls}) AUC={roc_auc_log[i]:.2f}')
        plt.plot(fpr_rf[i], tpr_rf[i], linestyle='--', label=f'RandForest (Class {cls}) AUC={roc_auc_rf[i]:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curves – Logistic Regression vs Random Forest')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.legend(); plt.show()

    # --- Feature Importance ---
    log_importance = pd.DataFrame({'Feature': FEATURES, 'Importance': np.abs(best_log_model.coef_[0])})
    rf_importance = pd.DataFrame({'Feature': FEATURES, 'Importance': best_rf_model.feature_importances_})
    print("\n=== Logistic Regression Feature Importance ===\n", log_importance.sort_values(by='Importance', ascending=False))
    print("\n=== Random Forest Feature Importance ===\n", rf_importance.sort_values(by='Importance', ascending=False))

    plt.figure(figsize=(10,5))
    plt.barh(rf_importance['Feature'], rf_importance['Importance'], color='green')
    plt.gca().invert_yaxis()
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance Score"); plt.ylabel("Feature")
    plt.show()

# ------------------ PREDICT 2026 FINALISTS ------------------
def predict_2026(best_model, scaler=None):
    future_data = pd.DataFrame({
        'Team': ['Argentina', 'France', 'Brazil', 'England', 'Spain', 'Portugal', 'Germany', 'Netherlands'],
        'Rank Difference': [5, 4, 3, 6, 7, 8, 9, 10],
        'Points Difference': [150, 130, 120, 100, 90, 80, 70, 60],
        'Home Team Avg Age': [27, 26, 28, 25, 27, 27, 26, 27],
        'Away Team Avg Age': [26, 27, 27, 26, 27, 28, 27, 27],
        'Home Team Experience': [72, 70, 71, 68, 69, 67, 65, 66],
        'Away Team Experience': [70, 68, 70, 69, 68, 66, 64, 65],
        'Home Team Win Rate': [0.75, 0.73, 0.72, 0.70, 0.69, 0.68, 0.67, 0.66],
        'Away Team Win Rate': [0.70, 0.71, 0.69, 0.68, 0.67, 0.66, 0.65, 0.64]
    })

    X_future = future_data[FEATURES]
    if scaler:
        X_future_scaled = scaler.transform(X_future)
        predictions = best_model.predict(X_future_scaled)
    else:
        predictions = best_model.predict(X_future)

    future_data['Predicted_Result'] = predictions
    outcome_map = {1: "Home Win", 0: "Draw", -1: "Away Win"}
    future_data['Predicted_Label'] = future_data['Predicted_Result'].map(outcome_map)
    print("\n=== 2026 Simulation Predictions ===")
    print(future_data[['Team', 'Predicted_Label']])

    # Predict win probabilities if supported
    if hasattr(best_model, 'predict_proba') and 1 in best_model.classes_:
        probs = best_model.predict_proba(X_future)
        win_index = list(best_model.classes_).index(1)
        future_data['Win_Probability'] = probs[:, win_index]
        finalists = future_data.sort_values(by='Win_Probability', ascending=False).head(2)
        print("\n🏆 Predicted 2026 Finalists:")
        print(finalists[['Team', 'Win_Probability']])

# ------------------ CLI ------------------
def cli():
    df = None
    best_log_model, best_rf_model, scaler, X_test, y_test, X, y = (None, None, None, None, None, None, None)

    while True:
        print("\n=== FIFA Prediction App ===")
        print("1) Run scraper (web_scraper.py)")
        print("2) Load cleaned dataset")
        print("3) Train & tune models")
        print("4) Evaluate models & show plots")
        print("5) Predict 2026 finalists")
        print("6) Exit")
        choice = input("\nEnter choice (1-6): ")

        if choice == "1":
            run_scraper_cli()
        elif choice == "2":
            df = load_dataset()
        elif choice == "3":
            if df is not None:
                best_log_model, best_rf_model, scaler, X_test, y_test, X, y = train_models(df)
            else:
                print(" Load dataset first.")
        elif choice == "4":
            if best_log_model and best_rf_model:
                evaluate_models(best_log_model, best_rf_model, scaler, X_test, y_test, X, y)
            else:
                print(" Train models first.")
        elif choice == "5":
            if best_log_model and best_rf_model:
                print("\nPredicting using Logistic Regression:")
                predict_2026(best_log_model, scaler)
                print("\nPredicting using Random Forest:")
                predict_2026(best_rf_model)
            else:
                print(" Train models first.")
        elif choice == "6":
            print("Exiting...")
            break
        else:
            print(" Invalid choice. Try again.")

if __name__ == "__main__":
    cli()
