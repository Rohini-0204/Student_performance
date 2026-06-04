import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Task 2: Load and Explore Dataset
try:
    df = pd.read_csv('Pass-Fail Data.csv')
    print("--- Dataset Loaded Successfully ---")
except FileNotFoundError:
    print("Error: 'Pass-Fail Data.csv' file not found. Please ensure it is in the correct directory.")
    exit()

# Task 3: Preprocess Data
# Excluding student_id and selecting the relevant features
X = df[['attendance_pct', 'homework_pct', 'midterm_score', 'study_hours_per_week']]
y = df['pass']

# Split the data into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Task 4: Train ML Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy Check
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

# Make a Prediction for a New Student
print("--- Prediction for New Student ---")
# New student details: Attendance = 80%, Homework = 75%, Midterm = 70, Study hours = 6
new_student = np.array([[80, 75, 70, 6]])
result = model.predict(new_student)

if result[0] == 1:
    print("Result: The student will PASS. 🎉")
else:
    print("Result: The student will FAIL. ⚠️")