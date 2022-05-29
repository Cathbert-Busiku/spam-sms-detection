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

class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "test":
            module = "text_process"
        return super().find_class(module, name)

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


@app.route('/',methods=['GET'])
def home():
    
    return render_template('index.html', info = 'login')

  

@app.route('/predict',methods=['POST'])
def predict():
    

    info = ['loged']
	
    with open('model_v01.pkl', 'rb') as f:
        unpickler = MyCustomUnpickler(f)
        spam_model = unpickler.load()
	
    if request.method == 'POST':
        
        # model = open('spam_model.pkl','rb')
        # spam_model= joblib.load(model)
        
		
        message = request.form['message']
        data = [message]
        # vect = cv.transform(data).toarray()
        my_prediction = spam_model.predict(data)
    return render_template('index.html', info = info, message = message, prediction = my_prediction)



if __name__ == '__main__':
    with open('model_v01.pkl', 'rb') as f:  
            spam_model = pickle.load(f)
    app.run(debug=True)
   
