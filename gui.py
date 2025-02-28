import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import csv
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

training = pd.read_csv('./Data/training.csv')
testing = pd.read_csv('./Data/testing.csv')

# Extract features (symptoms) and target (prognosis)
cols = training.columns[:-1]
x = pd.get_dummies(training[cols])  # One-hot encoding for symptoms
y = training['prognosis']

# Encode target variable (prognosis)
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

# One-hot encoding for testing set
testing_x = pd.get_dummies(testing[cols])
testing_y = le.transform(testing['prognosis'])

# Ensure column consistency between training and testing sets
x, testing_x = x.align(testing_x, join='left', axis=1, fill_value=0)

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Train Decision Tree Model
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

# Load additional symptom data
severity_dict = {}
description_dict = {}
precaution_dict = {}

def load_additional_data():
    global severity_dict, description_dict, precaution_dict

    with open('./Data/Symptom-severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if len(row) < 2:
                continue
            try:
                severity_dict[row[0]] = int(row[1])
            except ValueError:
                print(f"Warning: Could not convert severity for symptom '{row[0]}' to an integer.")

    with open('./Data/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if len(row) < 2:
                continue
            description_dict[row[0]] = row[1]

    with open('./Data/symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if len(row) < 5:
                continue
            precaution_dict[row[0]] = [row[1], row[2], row[3], row[4]]

load_additional_data()

# GUI Logic
current_question = 0
symptoms_exp = []
num_days = 0

questions = [
    "Enter your symptoms (comma-separated):",
    "How many days have you been experiencing these symptoms?",
    "Are you experiencing back pain? (yes/no)",
    "Are you experiencing weakness in limbs? (yes/no)",
    "Are you experiencing neck pain? (yes/no)",
    "Are you experiencing dizziness? (yes/no)",
    "Are you experiencing loss of balance? (yes/no)"
]

def predict_disease(symptoms_exp):
    input_vector = np.zeros(len(x.columns))
    for symptom in symptoms_exp:
        if symptom in x.columns:
            input_vector[x.columns.get_loc(symptom)] = 1
    prediction = clf.predict([input_vector])
    return le.inverse_transform(prediction)[0]

def on_submit():
    global current_question, symptoms_exp, num_days

    if current_question == 0:
        symptoms = entry_symptoms.get().split(',')
        symptoms = [symptom.strip() for symptom in symptoms]
        symptoms_exp.extend(symptoms)
        current_question += 1
        question_label.config(text=questions[current_question])
        entry_symptoms.delete(0, tk.END)

    elif current_question == 1:
        try:
            num_days = int(entry_symptoms.get())
            current_question += 1
            question_label.config(text=questions[current_question])
            entry_symptoms.delete(0, tk.END)
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid number.")

    elif current_question < len(questions):
        response = entry_symptoms.get().strip().lower()
        if response == "yes":
            symptoms_exp.append(questions[current_question].split(" ")[-1])
        current_question += 1
        if current_question < len(questions):
            question_label.config(text=questions[current_question])
            entry_symptoms.delete(0, tk.END)
        else:
            disease = predict_disease(symptoms_exp)
            description = description_dict.get(disease, "No description available.")
            precautions = precaution_dict.get(disease, ["No precautions available."])
            result_text = f"You may have: {disease}\n\nDescription: {description}\n\nPrecautions:\n" + "\n".join(precautions)
            messagebox.showinfo("Prediction Result", result_text)
            reset()

def reset():
    global current_question, symptoms_exp, num_days
    current_question = 0
    symptoms_exp = []
    num_days = 0
    question_label.config(text=questions[current_question])
    entry_symptoms.delete(0, tk.END)

# Create Tkinter GUI
root = tk.Tk()
root.title("HealthCare ChatBot")

question_label = tk.Label(root, text=questions[current_question])
question_label.pack()

entry_symptoms = tk.Entry(root, width=50)
entry_symptoms.pack()

submit_button = tk.Button(root, text="Submit", command=on_submit)
submit_button.pack()

root.mainloop()
