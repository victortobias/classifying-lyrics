# -*- coding: utf-8 -*-
from flask import Flask,render_template,url_for,request
import pandas as pd
import numpy as np
import nltk
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.externals import joblib

def clean_text(text):
    """
    Applies some pre-processing on the given text.

    Steps :
    - Removing HTML tags
    - Removing punctuation
    - Lowering text
    """
    
    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)    
    text = re.sub(r"\'", "", text)    
    text = re.sub(r"\"", "", text)    
    
    # convert text to lowercase
    text = text.strip().lower()
    
    # replace punctuation characters with spaces
    filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    return text

filename = 'modelmusic.sav' #nome do modelo treinado
cfl = joblib.load(filename)
stopwords = nltk.corpus.stopwords.words('portuguese')
cv = TfidfVectorizer(stop_words=stopwords, preprocessor=clean_text)
df = pd.read_excel('static/dataset.xlsx')
X = df['lyric']
X = cv.fit_transform(X)

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html', name='victor', css=url_for('static', filename='style.css'))

@app.route('/predict', methods=['POST'])
def pred():
	music = request.form['music']
	Z = pd.Series(music)
	Z = cv.transform(Z)

	predict = cfl.predict(Z)
	predict = predict[0]
	map_genre = {1: 'Sertanejo', 2: 'Axé', 3: 'Rap', 4: 'Funk Brasileiro', 5: 'Gospel'}
	return map_genre[predict]


if __name__ == '__main__':
	app.run(host="0.0.0.0", port=80)
