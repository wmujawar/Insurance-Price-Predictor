import os
import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__, template_folder=os.path.join(os.getcwd(), 'templates'), static_folder=os.path.join(os.getcwd(), 'static'))

@app.route('/', methods=['GET'])
def home():
    html_file = os.path.join(os.getcwd(), 'templates', 'index.html')
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Read data
    data = request.get_json()
    
    response, status_code, headers = _validate_input(data)
    
    if status_code:
        return response, status_code, headers
    
    df = pd.DataFrame.from_dict(data, orient='index').T
    
    # Load model
    with open(os.path.join(os.getcwd(), 'output', 'model.pkl'), 'rb') as model_file:
        model = pickle.load(model_file)

    # Make prediction
    prediction = model.predict(df)
    
    # Return prediction
    return str(prediction), 200, {'Content-Type': 'text/plain'}



def _validate_input(data):
    # Check if "Age" is between 18 and 110
    if not (18 <= data['Age'] <= 110):
        return "Age must be between 18 and 110.", 400, {'Content-Type': 'text/plain'}

    # Check if the following fields are either 0 or 1
    binary_fields = [
        'Diabetes', 
        'BloodPressureProblems', 
        'AnyTransplants', 
        'AnyChronicDiseases', 
        'KnownAllergies', 
        'HistoryOfCancerInFamily'
    ]
    for field in binary_fields:
        if data.get(field) not in [0, 1]:
            return f"{field} must be either 0 or 1.", 400, {'Content-Type': 'text/plain'}

    # Check if "Height" and "Weight" are non-negative
    if data['Height'] < 0 or data['Weight'] < 0:
        return "Height and Weight must be non-negative.", 400, {'Content-Type': 'text/plain'}

    # Check if "NumberOfMajorSurgeries" is between 0 and 3
    if not (0 <= data['NumberOfMajorSurgeries'] <= 3):
        return "NumberOfMajorSurgeries must be between 0 and 3.", 400, {'Content-Type': 'text/plain'}
    
    return None, None, None


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')

