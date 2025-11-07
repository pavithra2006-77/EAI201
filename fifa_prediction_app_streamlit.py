# fifa_prediction_app_streamlit.py

import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup
import csv
import webbrowser

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, auc)

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
def run_scraper():
    st.subheader("Running NBA Teams Scraper...")
    url = "https://www.nba.com/teams"
    try:
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        teams = soup.find_all('div', class_='TeamFigure_tf__jA5HW')

        if not teams:
            st.warning("No teams found. Website structure may have changed.")
            return

        csv_file = os.path.join(BASE_DIR, 'nba_teams.csv')
        with open(csv_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Team Name', 'Logo URL', 'Profile Link', 'Stats Link', 'Schedule Link', 'Tickets Link'])

            for team in teams:
                name_tag = team.find('a', class_='TeamFigure_tfMainLink__OPLFu')
                team_name = name_tag.text.strip() if name_tag else 'N/A'
                logo_tag = team.find('img')
                logo_url = logo_tag['src'] if logo_tag else 'N/A'

                links_div = team.find('div', class_='TeamFigure_tfLinks__gwWFj')
                links = links_div.find_all('a') if links_div else []

                profile, stats, schedule, tickets = ['N/A'] * 4
                if len(links) >= 4:
                    profile = "https://www.nba.com" + links[0]['href']
                    stats = "https://www.nba.com" + links[1]['href']
                    schedule = "https://www.nba.com" + links[2]['href']
                    tickets = "https://www.nba.com" + links[3]['href']

                writer.writerow([team_name, logo_url, profile, stats, schedule, tickets])

        st.success(f"Data scraped successfully and saved to {csv_file}")
    except Exception as e:
        st.error(f"Scraper encountered an error: {e}")

# ------------------ LOAD DATASET ------------------
def load_dataset():
    if not os.path.exists(PATH_CLEANED):
        st.warning(f" Cleaned dataset not found at {PATH_CLEANED}")
        return None
    df = pd.read_csv(PATH_CLEANED)
    st.success(f" Cleaned dataset loaded successfully ({df.shape[0]} rows, {df.shape[1]} columns).")
    st.dataframe(df.head())
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

    # Logistic Regression
    log_params = {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs']}
    log_grid = GridSearchCV(LogisticRegression(max_iter=5000), log_params, cv=5, scoring='accuracy')
    log_grid.fit(X_train_scaled, y_train)
    best_log_model = log_grid.best_estimator_

    # Random Forest
    rf_params = {'n_estimators':[100,200], 'max_depth':[None,5], 'min_samples_split':[2,5], 'min_samples_leaf':[1,2]}
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='accuracy')
    rf_grid.fit(X_train, y_train)
    best_rf_model = rf_grid.best_estimator_

    st.success(" Models trained successfully.")
    st.write("Logistic Regression Best Params:", log_grid.best_params_)
    st.write("Random Forest Best Params:", rf_grid.best_params_)

    return best_log_model, best_rf_model, scaler, X_test, y_test, X, y

# ------------------ EVALUATE MODELS ------------------
def evaluate_models(best_log_model, best_rf_model, scaler, X_test, y_test, X, y):
    X_test_scaled = scaler.transform(X_test)
    y_pred_log = best_log_model.predict(X_test_scaled)
    y_prob_log = best_log_model.predict_proba(X_test_scaled)

    y_pred_rf = best_rf_model.predict(X_test)
    y_prob_rf = best_rf_model.predict_proba(X_test)

    # Metrics
    metrics = pd.DataFrame({
        "Metric": ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"],
        "Logistic Regression": [
            accuracy_score(y_test,y_pred_log),
            precision_score(y_test,y_pred_log,average='weighted'),
            recall_score(y_test,y_pred_log,average='weighted'),
            f1_score(y_test,y_pred_log,average='weighted'),
            roc_auc_score(y_test,y_prob_log,multi_class='ovr')
        ],
        "Random Forest": [
            accuracy_score(y_test,y_pred_rf),
            precision_score(y_test,y_pred_rf,average='weighted'),
            recall_score(y_test,y_pred_rf,average='weighted'),
            f1_score(y_test,y_pred_rf,average='weighted'),
            roc_auc_score(y_test,y_prob_rf,multi_class='ovr')
        ]
    })
    st.dataframe(metrics)

    # Confusion Matrices
    cm_log = confusion_matrix(y_test,y_pred_log)
    cm_rf = confusion_matrix(y_test,y_pred_rf)
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title("Logistic Regression CM")
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title("Random Forest CM")
    st.pyplot(fig)

    # Feature Importance
    log_importance = pd.DataFrame({'Feature':FEATURES,'Importance':np.abs(best_log_model.coef_[0])})
    rf_importance = pd.DataFrame({'Feature':FEATURES,'Importance':best_rf_model.feature_importances_})
    st.write("Logistic Regression Feature Importance", log_importance.sort_values(by='Importance',ascending=False))
    st.write("Random Forest Feature Importance", rf_importance.sort_values(by='Importance',ascending=False))

    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.barh(rf_importance['Feature'], rf_importance['Importance'], color='green')
    ax2.invert_yaxis()
    ax2.set_title("Random Forest Feature Importance")
    st.pyplot(fig2)

# ------------------ PREDICT 2026 FINALISTS ------------------
def predict_2026(best_model, scaler=None):
    future_data = pd.DataFrame({
        'Team': ['Argentina', 'France', 'Brazil', 'England', 'Spain', 'Portugal', 'Germany', 'Netherlands'],
        'Rank Difference': [5,4,3,6,7,8,9,10],
        'Points Difference': [150,130,120,100,90,80,70,60],
        'Home Team Avg Age': [27,26,28,25,27,27,26,27],
        'Away Team Avg Age': [26,27,27,26,27,28,27,27],
        'Home Team Experience': [72,70,71,68,69,67,65,66],
        'Away Team Experience': [70,68,70,69,68,66,64,65],
        'Home Team Win Rate': [0.75,0.73,0.72,0.70,0.69,0.68,0.67,0.66],
        'Away Team Win Rate': [0.70,0.71,0.69,0.68,0.67,0.66,0.65,0.64]
    })

    X_future = future_data[FEATURES]

    # Scale only if using Logistic Regression
    if scaler:
        X_future_scaled = scaler.transform(X_future)
        probs = best_model.predict_proba(X_future_scaled)[:, list(best_model.classes_).index(1)]
    else:
        probs = best_model.predict_proba(X_future)[:, list(best_model.classes_).index(1)]

    future_data['Win_Probability'] = probs
    finalists = future_data.sort_values(by='Win_Probability', ascending=False).head(2)

    st.subheader("Predicted 2026 Finalists")
    st.dataframe(finalists[['Team', 'Win_Probability']])



# ------------------ STREAMLIT APP ------------------
st.title(" FIFA 2026 Prediction App")

if st.button("Run Scraper"):
    run_scraper()

df = load_dataset()
if df is not None:
    if st.button("Train Models"):
        best_log_model, best_rf_model, scaler, X_test, y_test, X, y = train_models(df)
        st.session_state['log_model'] = best_log_model
        st.session_state['rf_model'] = best_rf_model
        st.session_state['scaler'] = scaler
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
        st.session_state['X'] = X
        st.session_state['y'] = y

    if 'log_model' in st.session_state:
        if st.button("Evaluate Models & Show Plots"):
            evaluate_models(st.session_state['log_model'], st.session_state['rf_model'],
                            st.session_state['scaler'], st.session_state['X_test'],
                            st.session_state['y_test'], st.session_state['X'],
                            st.session_state['y'])
        if st.button("Predict 2026 Finalists"):
            st.subheader("Predictions using Logistic Regression")
            predict_2026(st.session_state['log_model'], st.session_state['scaler'])
            st.subheader("Predictions using Random Forest")
            predict_2026(st.session_state['rf_model'])
