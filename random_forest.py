import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Load dataset
file_path = "revised_titanium_alloy_data.csv"  # Update path if needed
df = pd.read_csv(file_path)

# Features and target
X = df.drop(columns=["EM_GPa"])
y = df["EM_GPa"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Best RandomForest model from tuning
rf = RandomForestRegressor(
    n_estimators=200,
    min_samples_split=2,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

# Train the model
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Final RandomForest Model")
print(f"R^2 Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

