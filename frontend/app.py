import numpy as np
import joblib
from flask import Flask, render_template, request #Flask: Web framework for building the application., render_template: Renders HTML pages., request: Used to get data from forms submitted by the user.

app = Flask(__name__) #This initializes the Flask app. __name__ refers to the current module, and Flask uses it to determine the location of the application.

# Load the saved model and scaler
model = joblib.load('heart_disease_rf_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/') #This route renders i.e.generates the home page (index.html), where the user will input their details
def index():
    return render_template('index.html')

# @app.route('/page2', methods=['POST']): This decorator tells Flask that when a POST request is made to the /page2 URL (i.e., when a form is submitted to this URL), the page2 function should be executed.
@app.route('/page2', methods=['POST']) #After the user submits the form on the home page, this route is triggered. The form data (e.g., name and age) is received from the user, and the page2.html template is rendered, showing the user's information.
def page2():
    name = request.form['name']
    age = int(request.form['age'])
    return render_template('page2.html', name=name, age=age)

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    age = int(request.form['age'])
    sex = int(request.form['sex'])  # Male = 1, Female = 0
    cholesterol = int(request.form['cholesterol'])
    blood_pressure = int(request.form['blood_pressure'])
    cp = int(request.form['cp'])
    trestbps = int(request.form['trestbps'])
    fbs = int(request.form['fbs'])
    restecg = int(request.form['restecg'])
    thalach = int(request.form['thalach'])
    exang = int(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = int(request.form['slope'])
    ca = int(request.form['ca'])
    thal = int(request.form['thal'])

    # Prepare input data for prediction (features from the form)
    prediction_input = np.array([[age, sex, cp, trestbps, cholesterol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

    # Scale the input data using the scaler
    prediction_input_scaled = scaler.transform(prediction_input)

    # Predict the heart disease risk
    prediction = model.predict(prediction_input_scaled)
    prediction_probability = model.predict_proba(prediction_input_scaled)[:, 1][0] * 100  # Convert to percentage

    # Round the prediction percentage to 2 decimal places
    prediction_percentage = round(prediction_probability, 2)

    # Return the prediction result to page2.html with prediction_percentage
    return render_template('page2.html', name=name, age=age, sex=sex, cholesterol=cholesterol, blood_pressure=blood_pressure, 
                           cp=cp, trestbps=trestbps, fbs=fbs, restecg=restecg, thalach=thalach, exang=exang, oldpeak=oldpeak, 
                           slope=slope, ca=ca, thal=thal, prediction_percentage=prediction_percentage)

@app.route('/recommendations', methods=['GET'])
def recommendations():
    # Get the prediction percentage and handle potential issues with missing values
    try:
        prediction_percentage = float(request.args.get('prediction_percentage', 50))
    except ValueError:
        prediction_percentage = 50

    # Create a recommendation message based on the prediction
    if prediction_percentage >= 75:
        message = "You have a high risk of heart disease. It's important to seek immediate medical attention and adopt a healthy lifestyle."
    elif prediction_percentage >= 50:
        message = "You have a moderate risk of heart disease. Consider consulting with a healthcare provider for further evaluation."
    else:
        message = "You have a low risk of heart disease. Keep maintaining a healthy lifestyle!"

    return render_template('page3.html', prediction_percentage=prediction_percentage, message=message)

# This starts the Flask development server when the script is run directly. The app will run in debug mode (useful for development).
if __name__ == '__main__':
    app.run(debug=True)
