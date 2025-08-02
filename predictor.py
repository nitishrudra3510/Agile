import joblib
import numpy as np

# Load the model
model = joblib.load("model/sprint_success_model.pkl")

print("Enter Sprint Details for Prediction:")


# Input from user
total_tickets = int(input("Total tickets in sprint: "))
bug_ratio = float(input("Bug-to-total ticket ratio (0 to 1): "))
avg_story_points = float(input("Average story points per ticket: "))
team_velocity = int(input("Team velocity: "))

# Predict
features = np.array([[total_tickets, bug_ratio, avg_story_points, team_velocity]])
prediction = model.predict(features)

if prediction[0] == 1:
    print("✅ This sprint is likely to be successful.")
else:
    print("⚠️ This sprint is likely to fail. Consider adjusting the scope or team load.")
