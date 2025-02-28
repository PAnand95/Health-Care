import numpy as np
import pandas as pd
import joblib

df1 = pd.read_csv("data/Symptom-severity.csv")
discrp = pd.read_csv("data/symptom_Description.csv")
precautions = pd.read_csv("data/symptom_precaution.csv")

rf_model = joblib.load("data/random_forest_model.joblib")
dt_model = joblib.load("data/decision_tree_model.joblib")

def predict_disease(model, symptoms_list):
    symptom_weights = dict(zip(df1["Symptom"].str.strip(), df1["weight"]))
    input_vector = [symptom_weights.get(symptom.strip(), 0) for symptom in symptoms_list]

    max_symptoms = 17
    input_vector += [0] * (max_symptoms - len(input_vector))
    input_vector = np.array(input_vector).reshape(1, -1)

    predicted_disease = model.predict(input_vector)[0].strip()

    # Validate disease presence
    if predicted_disease not in discrp["Disease"].values:
        print(f"\nPredicted Disease: {predicted_disease} (No description available)")
    else:
        disease_description = discrp[discrp["Disease"] == predicted_disease]["Description"].values[0]
        print(f"\nPredicted Disease: {predicted_disease}")
        print(f"Description: {disease_description}")

    # Get precautions if available
    if predicted_disease in precautions["Disease"].values:
        recommended_precautions = precautions[precautions["Disease"] == predicted_disease].iloc[:, 1:].values.flatten()
        print("Recommended Precautions:")
        for precaution in recommended_precautions:
            print(f"- {precaution}")
    else:
        print("No recommended precautions available.")

# Example Test
user_input = input("Enter symptoms separated by commas: ").split(",")
user_symptoms = [symptom.strip() for symptom in user_input]

print("\nUsing Decision Tree Model:")
predict_disease(dt_model, user_symptoms)

print("\nUsing Random Forest Model:")
predict_disease(rf_model, user_symptoms)
