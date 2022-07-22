import streamlit as st
import nltk
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer  
ps = PorterStemmer()
import pandas as pd
import numpy as np
import string
import pickle

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('Email/SMS Spam Classifier')

input_sms = st.text_input('Enter the message')

if st.button('Predict'):  #only if this button is clicked then only following steps will be performed

    # Step 1 - We have to do preprocessing
    def transform_text(text):
        
        # converting to lower case
        text = text.lower()
        
        
        # Tokenization
        text = nltk.word_tokenize(text)
        
        
        # Removing special characters ($,#,etc we we keep only aplhanumeric characters)
        y = []
        for i in text:
            if i.isalnum():  
                y.append(i)
        # 'Hi how% $200 Are You' becomes ['hi', 'how', '200', 'are', 'you']
        text = y[:]  # if we do text = y then the will point to same location and hence if we clear y again then text also gets cleared
        
        
        
        # Removing stop words and punctuations (technically puntuations taken care in above step but better to be explicit)
        #(Stop words are words that appear in high frequency in sentences but dont necessarily add much meaning/context)
        y = []
        for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(i) 
        #'Hi, Did you Like my $2000 coat I wore to the %%ML conference??' becomes ['hi', 'like', '2000', 'coat', 'wore', 'ml', 'conference']
        text = y[:]  #cloning
        
        
        
        # Stemming (getting the root form of words so that words like love,loving,loved are not treated as separate entitites)
        y = []
        for i in text:
            y.append(ps.stem(i))
        # 'I loved youtube Lectures on Machine LEarning, How about you??' becomes ['love', 'youtub', 'lectur', 'machin', 'learn']
        text = y[:]
        
        
        return " ".join(text)


    transformed_sms = transform_text(input_sms)

    # Step 2 - We have to vectorize the input

    vector_input = tfidf.transform([transformed_sms])

    # Step 3 - We have to predict

    result = model.predict(vector_input)[0]

    # Step 4 - We have to display the output

    if result == 1:
        st.header('SPAM')
    else:
        st.header('NOT SPAM')