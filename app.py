from flask import Flask, render_template, request
from keras.models import load_model
import sklearn
import pandas as pd
import tensorflow as tf
import pickle
import numpy as np


app = Flask(__name__)
# Load model from file
model_ann = load_model('ann.h5')
model_svr = pickle.load(open('svr.pkl','rb'))

# load data
data = pd.read_csv('data_without_norm.csv')

# define function to get previous year enrollments
def get_previous_enrollments(school_name):
    # get data for selected school
    school_data = data[data['Nama Sekolah'] == school_name]
    # print(school_data)
    # get enrollments for years 2016-2020
    enrollments = school_data.iloc[:, 1:].values.tolist()[0]
    # print(enrollments)
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
# Define the prediction route
@app.route('/predict-ann', methods=['GET','POST'])
def predict_ann():
    # get user input
    school_name = request.form['school']
    print('Nama sekolah yang dipilih : ',school_name)
    # get previous enrollments
    enrollments = get_previous_enrollments(school_name)
    print('Jumlah pendaftar tahun sebelumnya',enrollments)
    # make prediction for 2023
    prediction = model_ann.predict([enrollments])[0][0]
    print('Hasil prediksi',prediction)
    # round prediction to nearest integer
    prediction = int(prediction)
    # return prediction to user
    return render_template('svr-view.html', schools=data['Nama Sekolah'].unique().tolist(), prediction=prediction, school=school_name)

# Define the prediction route
@app.route('/predict-svr', methods=['GET','POST'])
def predict_svr():
    # get user input
    school_name = request.form['school']
    print('Nama sekolah yang dipilih : ',school_name)
    # get previous enrollments
    enrollments = np.array(get_previous_enrollments(school_name))
    print('Jumlah pendaftar tahun sebelumnya',enrollments)
    # make prediction for 2023
    prediction = model_svr.predict(enrollments.reshape(1, -1))[0]
    print('Hasil prediksi',prediction)
    # round prediction to nearest integer
    prediction = int(prediction)
    # return prediction to user
    return render_template('svr-view.html', schools=data['Nama Sekolah'].unique().tolist(), prediction=prediction, school=school_name)


if __name__ == '__main__':
    app.run(debug=True)