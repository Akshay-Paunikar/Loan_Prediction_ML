import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle

from sklearn.preprocessing import StandardScaler
from project.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    
    else:
        data = CustomData(
            Education = request.form.get('Education'),
            ApplicantIncome = float(request.form.get('ApplicantIncome')),
            CoapplicantIncome = float(request.form.get('CoapplicantIncome')),
            LoanAmount = float(request.form.get('LoanAmount')),
            Property_Area = request.form.get('Property_Area'),
            Gender = request.form.get('Gender'),
            Married = request.form.get('Married'),
            Dependents = request.form.get('Dependents'),
            Self_Employed = request.form.get('Self_Employed'),
            Loan_Amount_Term = float(request.form.get('Loan_Amount_Term')),
            Credit_History = float(request.form.get('Credit_History'))            
        )
        
        pred_df = data.get_data_as_data_frame()
        
        preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))
        
        pred_df = preprocessor.transform(pred_df)
        results = model.predict(pred_df)
        
        if results == 0.0:
            loan_status = "Rejected"
        else:
            loan_status = "Approved"       
        return render_template('index.html', results=loan_status)
    
if __name__ == "__main__":
    app.run(debug=True)