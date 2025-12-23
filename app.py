from flask import Flask,request, render_template

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData, Prediction_pipeline


application = Flask(__name__)

app = application

@app.route('/')

def index():
    return render_template('index.html' )

@app.route("/predictdata" , methods  = ['GET','POST'])

def predict_datapoint():
    if request.method == 'GET':
        return render_template("home.html")
    else:
        data = CustomData(
            Age = request.form.get('Age'),
            Sex = request.form.get('Sex'),
            ChestPainType = request.form.get('ChestPainType'),
            RestingBP = request.form.get('RestingBP'),
            Cholesterol = request.form.get('Cholesterol'),
            FastingBS = request.form.get('FastingBS'),
            RestingECG = request.form.get('RestingECG'),
            MaxHR = request.form.get('MaxHR'),
            ExerciseAngina = request.form.get('ExerciseAngina'),
            Oldpeak = request.form.get('Oldpeak'),
            ST_Slope = request.form.get('ST_Slope')
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = Prediction_pipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results)
    

if  __name__ == "__main__":
    app.run(host = '0.0.0.0', debug = True)