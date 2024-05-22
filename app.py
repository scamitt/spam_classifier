import streamlit as st
import pickle
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

file1 = open('vectorizer.pkl', 'rb')
tfidf = pickle.load(file1)
file2 = open('model.pkl', 'rb')
model = pickle.load(file2)
ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)

    y = []
    for token in text:
        if token.isalnum():
            y.append(token)

    text = y[:]
    y.clear()

    for token in text:
        if token not in stopwords.words('english') and token not in string.punctuation:
            y.append(token)

    text = y[:]
    y.clear()

    for token in text:
        y.append(ps.stem(token))

    ret = ' '.join(y)
    return ret


st.title('SMS Spam Classifier')
sms = st.text_input('Enter your message')
if st.button('Predict'):
    transform_sms = transform_text(sms)
    if transform_sms:  # Check if the user has entered any message
        vector = tfidf.transform([transform_sms])

        result = model.predict(vector)[0]

        if result == 0:
            st.header('Not Spam')
        else:
            st.header('Spam')
