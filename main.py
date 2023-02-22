from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
import spacy
import fastai
from fastai.text import *


app = Flask("__name__")

q = ""

@app.route("/")
def loadPage():
	return render_template('home.html', query="")



@app.route("/predict", methods=['POST'])
def predict():
    
    lm = load_learner("models", 'exported_lm.pkl')
    

    input_lyric = request.form['query1']
    number_of_generated_words = request.form['query2']
    # inputQuery3 = request.form['query3']
    # inputQuery4 = request.form['query4']
    # inputQuery5 = request.form['query5']

    # https://dylan-lyrics.uw.r.appspot.com
    
    o2 = ""
    
    generated_lyrics = lm.predict(input_lyric, n_words=int(number_of_generated_words))
    #print('data is: ')
    #print(data)
    #016.14, 74.00, 0.01968, 0.05914, 0.1619
    
    # Create the pandas DataFrame     
    
    return render_template('home.html', output1=generated_lyrics, output2=o2, query1 = request.form['query1'], query2 = request.form['query2'])
    
if __name__ == "__main__":
    app.run(debug=True)
