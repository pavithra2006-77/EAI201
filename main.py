import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import os

# --- 0. LOAD CLEANED DATASET ---
base_dir = r"C:\Users\test\OneDrive\Desktop\assignment2aiml"
cleaned_file = 'cleaned_fifa_dataset.csv'
path_cleaned = os.path.join(base_dir, cleaned_file)

df_master = pd.read_csv(path_cleaned)
print("Cleaned dataset loaded successfully.")

# --- 1. SELECT FEATURES & TARGET ---
features = ['Rank Difference', 'Points Difference',
            'Home Team Avg Age', 'Away Team Avg Age',
            'Home Team Experience', 'Away Team Experience',
            'Home Team Win Rate', 'Away Team Win Rate']
target = 'Home_Win'

X = df_master[features]
Y = df_master[target]

# --- 2. SPLIT DATASET ---
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# --- 3. SCALE FEATURES ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4. TRAIN LOGISTIC REGRESSION ---
model = LogisticRegression(solver='lbfgs', max_iter=5000)
model.fit(X_train_scaled, Y_train)

# --- 5. EVALUATE MODEL ---
Y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(Y_test, Y_pred)

print("\n*** MODEL EVALUATION RESULTS ***")
print(f"Features Used: {features}")
print(f"Accuracy on Test Set: {accuracy * 100:.2f}%")

# --- 6. EXAMPLE PREDICTION ---
example_data = pd.DataFrame({
    'Rank Difference': [10],
    'Points Difference': [150],
    'Home Team Avg Age': [27],
    'Away Team Avg Age': [28],
    'Home Team Experience': [70],
    'Away Team Experience': [72],
    'Home Team Win Rate': [0.6],
    'Away Team Win Rate': [0.55]
})

example_scaled = scaler.transform(example_data)
prediction = model.predict(example_scaled)[0]
outcome_map = {1: "Home Win", 0: "Draw", -1: "Away Win"}
print(f"\nExample Prediction (Stronger Home Team): {outcome_map[prediction]}")
