from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np

# Your API definition
app = Flask(__name__)



@app.route('/churn/predict', methods=['POST'])
def churnpredict():
    model1 = joblib.load("randomforestchurnmodel.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')

    if model1:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)
            
            prediction = model1.predict_proba(query)
           
            output = prediction[:,0]*100
            a_list = list(output)
            print(a_list[0])
            a= round(a_list[0], 2) 
            return jsonify({'The probability of a customer becoming defaulter is': str(a)})
            

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')


    
@app.route('/homeloan/predict', methods=['POST'])
def homeloanpredict():
    model2= joblib.load("logistic_regression_homeloan_model.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("homeloanmodel_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')
    if model2:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)
            
            prediction = model2.predict_proba(query)
           
            output = prediction[:,1]*100
            a_list = list(output)
            print(a_list[0])
            a= round(a_list[0], 2) 
            return jsonify({'The probability of a customer getting home loan ': str(a)})
            

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')


    
@app.route('/personalloan/predict',  methods=['POST'])
def personalloanpredict():
    model3 = joblib.load("logistic_regression_personalloan.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("personalloanmodel_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')
    if model3:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)
            
            prediction = model3.predict_proba(query)
           
            output = prediction[:,1]*100
            a_list = list(output)
            print(a_list[0])
            a= round(a_list[0], 2) 
            return jsonify({'The probability of a customer getting personal  loan ': str(a)})
            

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')


if __name__ == '__main__':
   app.run("0.0.0.0",port=5000, debug=True)

    
    
    
   
