from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load trained model
model = joblib.load("crop_model.pkl")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None

    if request.method == 'POST':
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        data = pd.DataFrame({
            'Temperature': [temperature],
            'Humidity': [humidity],
            'pH': [ph],
            'Rainfall': [rainfall]
        })

        prediction = model.predict(data)[0]

    return render_template('predict.html', prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)