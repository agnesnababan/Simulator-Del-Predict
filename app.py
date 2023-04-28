from flask import Flask, render_template, request
from keras.models import load_model
import sklearn
import pandas as pd
import tensorflow as tf
import pickle

app = Flask(__name__)
# Load model from file
model = load_model('ann.h5')
# load data
data = pd.read_csv('data_without_norm.csv')

# Load svr model from file
with open('svr.pkl', 'rb') as f:
    svr_model = pickle.load(f)

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
    school = data['Nama Sekolah'].unique().tolist()
    return render_template('index.html', school=school)
# Define the svr route
@app.route('/svr')
def svr():
    return render_template('svr-view.html')
# Define the ann route
@app.route('/ann')
def ann():
    return render_template('ann-view.html')

# Define the prediction route
@app.route('/predict', methods=['GET','POST'])
def predict():
    # get user input
    school_name = request.form['school']
    print('Nama sekolah yang dipilih : ',school_name)
    # get previous enrollments
    enrollments = get_previous_enrollments(school_name)
    print('Jumlah pendaftar tahun sebelumnya',enrollments)
    # make prediction using ANN model
    ann_prediction = model.predict([enrollments])[0][0]
    print('Hasil prediksi ANN',ann_prediction)
    # make prediction using SVR model
    svr_prediction = svr_model.predict([enrollments])[0]
    print('Hasil prediksi SVR',svr_prediction)
    # round prediction to nearest integer
    ann_prediction = int(ann_prediction)
    svr_prediction = int(svr_prediction)
    # return prediction to user
    return render_template('svr-view.html', schools=data['Nama Sekolah'].unique().tolist(), 
                           ann_prediction=ann_prediction, svr_prediction=svr_prediction, 
                           school=school_name)

if __name__ == '__main__':
    app.run(debug=True)