import sys
import pandas as pd
from exception import CustomException
from utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor =  load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,
                 Education:str,
                 ApplicantIncome:int,
                 CoapplicantIncome:int,
                 LoanAmount:int,
                 Property_Area:str,
                 Gender:str,
                 Married:str,
                 Dependents:str,
                 Self_Employed:str,
                 Loan_Amount_Term:str,
                 Credit_History:str):
        
        self.Education = Education
        self.ApplicantIncome = ApplicantIncome
        self.CoapplicantIncome = CoapplicantIncome
        self.LoanAmount = LoanAmount
        self.Property_Area = Property_Area
        self.Gender = Gender
        self.Married = Married
        self.Dependents = Dependents
        self.Self_Employed = Self_Employed
        self.Loan_Amount_Term = Loan_Amount_Term
        self.Credit_History = Credit_History
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Education":[self.Education],
                "ApplicantIncome":[self.ApplicantIncome],
                "CoapplicantIncome":[self.CoapplicantIncome],
                "LoanAmount":[self.LoanAmount],
                "Property_Area":[self.Property_Area],
                "Gender":[self.Gender],
                "Married":[self.Married],
                "Dependents":[self.Dependents],
                "Self_Employed":[self.Self_Employed],
                "Loan_Amount_Term":[self.Loan_Amount_Term],
                "Credit_History":[self.Credit_History]
            }
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)
        