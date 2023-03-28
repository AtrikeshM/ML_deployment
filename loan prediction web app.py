# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 20:35:25 2023

@author: Hp
"""

import numpy as np
import pickle
import streamlit as st

#Loading the saved module
loaded_model = pickle.load(open('C:/Users/Hp/OneDrive/Desktop/ML_Deployment/trained_model.sav','rb'))

      
def loan_predict(input_data):
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
    
    # changing the input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the np array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    # print(prediction)

    if (prediction[0]=='Y'):
      return 'You can get a loan'
    else:
      return 'You are not eligible to get a loan'
  
    
def main():
    
    #giving title
    st.title('Loan Prediction System')
    
    #getting the data from user
    
    #Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,
    #CoapplicantIncome,LoanAmount,
    #Loan_Amount_Term,Credit_History,Property_Area
      
    Gender = st.number_input('Enter your gender')
    Married = st.number_input('Are you Married')
    Dependents = st.number_input('No. of Dependents')
    Education = st.number_input('Are you educated')
    Self_Employed = st.number_input('Are you Self_Employed')
    ApplicantIncome = st.number_input('Enter your ApplicantIncome')
    CoapplicantIncome = st.number_input('Enter your CoapplicantIncome')
    LoanAmount = st.number_input('Enter your LoanAmount')
    Loan_Amount_Term = st.number_input('Enter your Loan_Amount_Term')
    Credit_History = st.number_input('Enter your Credit_History')
    Property_Area = st.number_input('Enter your Property_Area')
    
    
    #code for prediction
    
    result = ''
    
    # creating a button for prediction
    
    if st.button('Check your ability'):
        result = loan_predict([Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area])
    
    
    st.success(result)
    
    
if __name__ == '__main__':
    main()