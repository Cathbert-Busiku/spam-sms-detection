from flask import Flask,render_template,url_for,request
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import joblib
import string
import nltk
from nltk.corpus import stopwords


app = Flask(__name__)

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    # messages= pd.read_csv("messages.csv")
    # from sklearn.model_selection import train_test_split
    # msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.2)
    #cathbert busiku
    # from sklearn.pipeline import Pipeline

    # pipeline = Pipeline([
    # ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    # ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    # ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
    # ])

    # pipeline.fit(msg_train,label_train)

    # predictions = pipeline.predict(msg_test)


    # df= pd.read_csv("spam.csv", encoding="latin-1")
    # df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    # # Features and Labels
    # df['label'] = df['class'].map({'ham': 0, 'spam': 1})
    # X = df['message']
    # y = df['label']
    
    # # Extract Feature With CountVectorizer
    #cv = CountVectorizer()
    # X = cv.fit_transform(X) # Fit the Data
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # #Naive Bayes Classifier
    # from sklearn.naive_bayes import MultinomialNB

    # clf = MultinomialNB()
    # clf.fit(X_train,y_train)
    # clf.score(X_test,y_test)

    # Alternative Usage of Saved Model
    # joblib.dump(clf, 'NB_spam_model.pkl')
    model = open('spam_model.pkl','rb')
    spam_model= joblib.load(model)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        # vect = cv.transform(data).toarray()
        my_prediction = spam_model.predict(data)
    return render_template('index.html',message = message, prediction = my_prediction)



if __name__ == '__main__':
    app.run(debug=True)