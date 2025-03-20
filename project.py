

from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)  

model = joblib.load("symptom_checker_model.pkl")
dataset_path = r"C:\Symptom Checker\symbipredict_2022.csv"
df = pd.read_csv(dataset_path)
symptom_columns = df.drop(columns=["prognosis"]).columns.tolist()
label_encoder = joblib.load("label_encoder.pkl")

def predict_disease(symptom_list):
    input_data = np.zeros(len(symptom_columns))
    for symptom in symptom_list:
        if symptom in symptom_columns:
            input_data[symptom_columns.index(symptom)] = 1  
    prediction = model.predict([input_data])
    predicted_disease = label_encoder.inverse_transform(prediction)[0]
    return predicted_disease

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  
    symptoms = data.get("symptoms", [])
    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400  
    predicted_disease = predict_disease(symptoms)
    return jsonify({"predicted_disease": predicted_disease})

if __name__ == "__main__":
    app.run(debug=True)
