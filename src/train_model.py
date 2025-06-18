import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import os
import joblib
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from utils import preprocess_data

# Load the cleaned dataset
df_cleaned = pd.read_csv("data/raw/cleaned_data.csv")

# Preprocess the data
X, y, scaler = preprocess_data(df_cleaned)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):,.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

# Save model and scaler
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/random_forest_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')