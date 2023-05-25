from flask import Flask, render_template, request
from keras.models import load_model
import sklearn
import pandas as pd
import tensorflow as tf
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler


app = Flask(__name__)
# Load model from file
model_ann = load_model('ann.h5')
model_svr = pickle.load(open('svr.pkl','rb'))
scaler = StandardScaler()

# load data
data = pd.read_csv('data_without_norm.csv')

# define function to get previous year enrollments
def get_previous_enrollments(school_name):
    # get data for selected school
    school_data = data[data['Nama Sekolah'] == school_name]
    # print(school_data)
    # get enrollments for years 2016-2022
    enrollment = school_data.iloc[:, 1:].values.reshape(-1,1)
    print('ini enrollment: ', enrollment)
    enrollments=scaler.fit_transform(enrollment)
    print('Data sekolah yang di normalisasi: ',enrollments)
    return enrollments

# Define the home route
@app.route('/')
def home():
    # get unique school names
    # school = data['Nama Sekolah'].unique().tolist()
    return render_template('index.html')

# Define the svr route
@app.route('/svr')
def svr():
    # get unique school names
    school = data['Nama Sekolah'].unique().tolist()
    return render_template('svr-view.html', school=school)
# Define the ann route
@app.route('/ann')
def ann():
    # get unique school names
    school = data['Nama Sekolah'].unique().tolist()
    return render_template('ann-view.html', school=school)
# # Define the prediction route
# @app.route('/predict-ann', methods=['GET','POST'])
# def predict_ann():
#     # get user input
#     school_name = request.form['school']
#     print('Nama sekolah yang dipilih : ',school_name)
#     # get previous enrollments
#     enrollments = get_previous_enrollments(school_name)
#     print('Jumlah pendaftar tahun sebelumnya',enrollments)
#     # make prediction_ann for 2023
#     prediction_ann = model_ann.predict(enrollments.reshape(1, -1))[0]
#     #inverse transform the prediction
#     prediction_ann = scaler.inverse_transform(prediction_ann.reshape(-1, 1))
#     print('Hasil prediksi', prediction_ann)
#     #round the prediction to nearest integer and ensure it is positive
#     prediction_ann = int(abs(prediction_ann[0][0]))

#     return render_template('ann-view.html', schools=data['Nama Sekolah'].unique().tolist(), prediction_ann=prediction_ann, school=school_name, enrollments = enrollments)

# Define the prediction route
@app.route('/predict-ann', methods=['GET','POST'])
def predict_ann():
    scaler = MinMaxScaler()
    # get user input
    school_name = request.form['school']
    school_data = data[data['Nama Sekolah'] == school_name]
    X = data.iloc[:, 1:].values
    X= scaler.fit_transform(X)
    
    X_pred = school_data.iloc[:,1:].values
    
    X_forecast_norm = scaler.transform(X_pred)
    y_forecast = model_ann.predict(X_forecast_norm)
    #round the prediction to nearest integer and ensure it is positive
    y_forecast = int(y_forecast)
    return render_template('ann-view.html', prediction_ann=y_forecast, school_name=school_name, enrollments = X_forecast_norm)

# Define the prediction route
@app.route('/predict-svr', methods=['GET','POST'])
def predict_svr():
    scaler = MinMaxScaler()
    # get user input
    school_name = request.form['school']
    school_data = data[data['Nama Sekolah'] == school_name]
    X = data.iloc[:, 1:].values
    X= scaler.fit_transform(X)
    
    X_pred = school_data.iloc[:,1:].values
    
    X_forecast_norm = scaler.transform(X_pred)
    y_forecast = model_svr.predict(X_forecast_norm)
    #round the prediction to nearest integer and ensure it is positive
    y_forecast = int(y_forecast)
    return render_template('svr-view.html', prediction_svr=y_forecast, school_name=school_name, enrollments = X_forecast_norm)


if __name__ == '__main__':
    app.run(debug=True)