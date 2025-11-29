import streamlit as st
import pickle
import os
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(nltk_data_path):
    nltk.download("punkt_tab")
    nltk.download('stopwords')
ps = PorterStemmer()

def text_transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalpha():
            y.append(i)
    
    text = y.copy()
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y.copy()
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

tfid = pickle.load(open(r"vectorizer.pkl",'rb'))
model = pickle.load(open(r"model.pkl",'rb'))

st.title("Email and SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):

    transformed_sms = text_transform(input_sms)

    vector_input = tfid.transform([transformed_sms])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("spam")
    else:
        st.header("Not Spam")