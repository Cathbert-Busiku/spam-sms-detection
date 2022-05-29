from flask import Flask,render_template,url_for,request
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import joblib
import string
import nltk
from nltk.corpus import stopwords
import pickle
from test import text_process




app = Flask(__name__)





@app.route('/',methods=['GET'])
def home():
    
    return render_template('index.html', info = 'login')




    

@app.route('/predict',methods=['POST'])
def predict():
    

    info = ['loged']

    if request.method == 'POST':
        
        model = open('spam_model.pkl','rb')
        spam_model= joblib.load(model)
		
        message = request.form['message']
        data = [message]
        # vect = cv.transform(data).toarray()
        my_prediction = spam_model.predict(data)
    return render_template('index.html', info = info, message = message, prediction = my_prediction)



if __name__ == '__main__':
    app.run(debug=True)
   