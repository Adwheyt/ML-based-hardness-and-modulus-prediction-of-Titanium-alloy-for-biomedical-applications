import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

# Load dataset
file_path = "revised_titanium_alloy_data.csv"  # Update path if needed
df = pd.read_csv(file_path)

# Features and target
X = df.drop(columns=["EM_GPa"])
y = df["EM_GPa"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# GradientBoosting with updated depth
best_gbr = GradientBoostingRegressor(
    subsample=0.9,
    n_estimators=1000,
    max_features=None,
    max_depth=6,  # Increased depth from 4 to 6
    learning_rate=0.005,
    random_state=42
)

# Train the model
best_gbr.fit(X_train, y_train)

# Predictions
y_pred = best_gbr.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("===== GradientBoostingRegressor with max_depth=6 =====")
print(f"R^2 Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

