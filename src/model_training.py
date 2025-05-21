import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Select features and target (example — adjust based on your real feature engineering)
X = df.drop(['Attrition'], axis=1)  # Replace with proper feature columns
y = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)  # Binary target

# Preprocessing example: encode categorical variables
X = pd.get_dummies(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "models/attrition_model.pkl")

# Save the feature columns so the app can align inputs
joblib.dump(X.columns.tolist(), "models/features.pkl")

print("Model and features saved successfully!")

