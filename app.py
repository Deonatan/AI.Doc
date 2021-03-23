from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('trainedResult.pkl','rb'))


@app.route('/')
def mainPageTemplate():
    return render_template('mainPage.html')

@app.route('/heartdisease')
def renderHeart():
    return render_template('heartdisease.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    predicted=model.predict([int_features])

    if predicted == 0:
        return render_template('heartdisease.html',pred='From your test result, you do not have heart disease indication')
    elif predicted == 1:
        return render_template('heartdisease.html',pred='From your test result, you have strong indicaton of heart disease.')
    else:
        return(render_template('heartdisease.html',pred='no result found'))



if __name__ == '__main__':
    app.run(debug=True)

