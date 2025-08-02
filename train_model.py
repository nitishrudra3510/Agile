import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv("data/synthetic_sprint_data.csv")

# Features and labels
X = data[['total_tickets', 'bug_ratio', 'avg_story_points', 'team_velocity']]
y = data['success']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

# Save model
joblib.dump(model, "model/sprint_success_model.pkl")
print("Model saved.")
