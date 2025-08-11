import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    # Lowercasing
    text = text.lower()

    # Tokenization
    text = nltk.word_tokenize(text)

    # Removing non-alphanumeric tokens
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    # Removing stopwords and punctuations
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    # Stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfdif = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")
if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tfdif.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
        st.header("This is Spam")
    else:
        st.header("This is Not Spam")