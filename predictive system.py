# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

#Loading the saved module
loaded_model = pickle.load(open('C:/Users/Hp/OneDrive/Desktop/ML_Deployment/trained_model.sav','rb'))

import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    #Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area

    input_data = (1,1,0,0,0,17,2840,114,360,0,0)



    # changing the input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the np array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    # print(prediction)

    if (prediction[0]=='Y'):
      print('You can get a loan')
    else:
      print('You are not eligible to get a loan')