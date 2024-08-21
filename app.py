import joblib
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer

model = joblib.load("Spam_Detection.h5")
cv = joblib.load("CountVectorizer.h5")
st.header("Spam Detection")

def predict(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result

input_message = st.text_input("Enter Message Here")

if st.button("Validate"):
    output = predict(input_message)
    st.markdown(output)