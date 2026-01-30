import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Initialize stemmer
ps = PorterStemmer()

# Load TF-IDF vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
             
    return " ".join(y)

# Streamlit UI
st.title("Email Spam Classifier")
input_sms = st.text_input("Enter the email content:")

if input_sms:
    # Preprocessing
    transformed_sms = transform_text(input_sms)

    # Vectorization
    vector_input = tfidf.transform([transformed_sms])

    # Prediction
    result = model.predict(vector_input)[0]

    # Display result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
