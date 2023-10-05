import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

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


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))



#set background

st.markdown(
    """
    <style>
     .stApp{
     background-image: url("https://uploads-ssl.webflow.com/61afa38359ee8f30af9023d3/628e6c689f45145f1a17dab5_Frame%204%20(2).png");
     background_size: cover;
     }
    </style>
    """,
    unsafe_allow_html=True
)
# set bigger heading
st.title("SMS Spam Classifier Machine Learning Model")


input_sms = st.text_area("Enter the message to check if the message is spam or not")
if st.button('Predict'):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
