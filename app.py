 # Importing essential libraries
from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
import pandas as pd
from datetime import datetime

# Load the LogisticRegression model
model = pickle.load(open('metro_intestate23.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('main.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        rain_1h = float(request.form['a'])
        snow_1h = float(request.form['b'])
        clouds_all = int(request.form['c'])    
        weather_main = request.form.get('d')
        weather_description = request.form.get('e')
        temp = float(request.form['h']) 
        holiday = request.form.get('f')       
        
        form_date = request.form.get('j')
        
      
        
        date_time_year =int(pd.to_datetime(form_date, format ="%Y-%m-%dT%H:%M").year)
        date_time_month=int(pd.to_datetime(form_date, format ="%Y-%m-%dT%H:%M").month)
        date_time_week=int(pd.to_datetime(form_date, format ="%Y-%m-%dT%H:%M").week)
        date_time_day=int(pd.to_datetime(form_date, format="%Y-%m-%dT%H:%M").day)
        date_time_hour=int(pd.to_datetime(form_date, format ="%Y-%m-%dT%H:%M").hour)
        date_time_dayofweek=int(pd.to_datetime(form_date, format ="%Y-%m-%dT%H:%M").dayofweek)
                    
        
        data = np.array([[holiday,temp,rain_1h,snow_1h,clouds_all,weather_main,weather_description,date_time_year,date_time_month,
        date_time_week,date_time_day,date_time_hour,date_time_dayofweek]])
        
        data.tofile('sample3.csv',sep=',')
        
        my_prediction = model.predict(data)
        
        a = np.array(my_prediction)
        lis = my_prediction.tolist()
        my_prediction = round(lis[0],2)
        
        #a.tofile('sample1.csv',sep=',')
        return render_template('result.html', prediction=my_prediction)
        

if __name__ == '__main__':
     app.run(debug=True)
