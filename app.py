from flask import Flask, request, jsonify, render_template, redirect, url_for
import numpy as np
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings

warnings.simplefilter('ignore')


csv_path = r'Heart-Disease-Predicition-main/Heart_Disease_Prediction.csv'
heart_df = pd.read_csv(csv_path)

encoder = LabelEncoder()
encoder.fit(heart_df['Heart Disease'])
heart_df['Heart Disease'] = encoder.transform(heart_df['Heart Disease'])


X = heart_df.drop(columns='Heart Disease', axis=1)
Y = heart_df['Heart Disease']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

model_path = 'heart_disease_model.pkl'
joblib.dump(model, model_path)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    return redirect(url_for('predict_form'))
    
@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/signup', methods=['POST'])
def register():
   
    return redirect(url_for('predict_form'))
@app.route('/predict_form')
def predict_form():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(f"Received data: {data}")  


        input_data = np.array([
            float(data['age']), int(data['sex']), int(data['chest_pain']), float(data['resting_bp']),
            float(data['cholesterol']), int(data['fasting_bs']), int(data['rest_ecg']), float(data['max_hr']),
            int(data['exercise_angina']), float(data['oldpeak']), int(data['slope']), int(data['ca']),
            int(data['thal'])
        ]).reshape(1, -1)
        print(f"Input data array: {input_data}")  

        model = joblib.load(model_path)
        prediction = model.predict(input_data)
        print(f"Prediction: {prediction}") 

        if prediction[0] == 1:
            result = 'Heart disease detected'
        else:
            result = 'No heart disease detected'

        return jsonify({'result': result})
    except Exception as e:
        print(f"Error during prediction: {e}")  
        return jsonify({'result': f"Error during prediction: {e}"}), 500
@app.route('/result')
def result():
    result = request.args.get('result', '')
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
