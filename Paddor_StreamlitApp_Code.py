import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from PIL import Image
import pickle
import spacy

#Name of App & pages
st.set_page_config(page_title="Sentiment Analyzer", layout="wide")
st.title("Sentiment Analyzer")
data_overview, model_making, model = st.tabs(["Problem & Data Overview","Model Creation", "Model Overview & Testing"])

with data_overview:
    #Creating Problem subheader with problem description and big story chart 
    st.subheader("Problem")
    "Physicians are looking for a model to predict the sentiment of patient drug prescription reviews."
    image = Image.open('Picture1.jpg')
    st.image(image, caption='Big Story Chart for Drug Prescription Sentiment Analysis Model')
    # getting data from your firestore database - reddit collection
    st.subheader("Data Overview")
    "The dataset provides patient reviews on various prescription drugs related to different health conditions. Ratings are available based on overall satisfaction"
    #Creating Data Display of ratings
    df = pd.read_csv("train (1).csv")
    st.subheader("Sample Posts")
    st.write(df.head(5))
    #Creating Histogram of ratings
    fig = px.histogram(df, x='rating', nbins=5)
    st.plotly_chart(fig)
    st.caption("number of reviews per rating")

    st.write("The majority of posts have a rating of 6 or higher, so the dataset is imbalanced!")
    #Explaining cleaning data
    st.subheader("Cleaning Data")
    st.write("To clean data, I labeled data with a rating of 4 and lower as negative and the rest as positive using a defined function.")
    rating = st.selectbox("Select a rating", ["1", "2","3", "4", "5", "6", "7", "8", "9", "10"])
    if rating == ["1", "2", "3", "4"]:
        st.writes("The rating is negative.")
    else:
        st.write("The rating is positive.")       
    st.write("Determining whether the rating is positive or negative helps train and validate the model by creating a binary prediction!")
    st.subheader("Splitting data into testing and training")
    st.write("data was split into two datasets for training and testing. 70% of the data was used for training, while 30% were used for testing.")
    st.write("I used an entirely different set of data to test the model. The data was structured the same way, but didn't need to be split up into two seperate sets")
with model_making:
    st.subheader("Models Tested")
    st.write("The model I used consisted of a pipeline that had two parts: 1) TfidfVectorizer, 2)LinearSVC")
    st.write("To tune my model, I tested five tokenizers in my pipeline. Click below to see descriptions.")
    #Downloading images of code
    tokenizer1 = Image.open("tokenizer1.jpg")
    tokenizer2 = Image.open("tokenizer2.jpg")
    tokenizer3 = Image.open("tokenizer3.jpg")
    tokenizer4 = Image.open("tokenizer4.jpg")
    tokenizer5 = Image.open("tokenizer5.jpg")
    #Writing selectbox to display different tokenizer descriptions
    tokenizer_desc = st.selectbox("Select a tokenizer", ["Tokenizer 1", "Tokenizer 2", "Tokenizer 3", "Tokenizer 4", "Tokenizer 5"])
    if tokenizer_desc == "Tokenizer 1":
        st.write("The first tokenizer was a replica of the spacy tokenizer used in class.")
        st.image(tokenizer1)
    elif tokenizer_desc == "Tokenizer 2":
        st.write("The second tokenizer was a basic spacey tokenizer that defined tokens.")
        st.image(tokenizer2)
    elif tokenizer_desc == "Tokenizer 3":
        st.write("The third model incorporated POS.")
        st.image(tokenizer3)
    elif tokenizer_desc == "Tokenizer 4":
        st.write("The fourth model includes POS and STOP_words.")
        st.image(tokenizer4)
    else:
        st.write("The fifth model uses POS, STOP_words, and Lexicons.")
        st.image(tokenizer5)


with model:
    st.subheader("Model Performance by tokenizer")
    #writing table to display accuracy and F1 scores
    st.write("Each model provided different accuracy and F1 scores for testing data and training data")
    st.write("Train & Test Dataset Scores")
    scores = {
    'Tokenizer': ['1', '2', '3', '4', '5'],
    'Train Accuracy Score': [77.63, 77.20, 76.23, 78.06, 76.989],
    'Train F1 Score': [86.85, 86.63, 86.03, 87.10, 86.63],
    'Test Accuracy Score': [88.99, 90.15, 88.41, 88.41, 87.54],
    'Test F1 Score': [94.17, 94.82, 93.85, 93.85, 93.36]
    }
    st.table(scores)
    st.write("Based on the testing scores, I chose tokenizer 2 as the most effective model!")
    st.subheader("Our Sentiment Model: Tokenizer 2")
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", "90.15%")
    col2.metric("F1 Score", "94.82")

    st.subheader("Test out the model")
    #load trained model
    nlp = spacy.load('en_core_web_sm')

    def spacy_tokenizer(doc):
      tokens = nlp(doc)
      return [token.text for token in tokens if not token.is_stop and not token.is_punct and not token.is_space]
    with open('pipeline (1).pkl','rb') as f:
      model = pickle.load(f)
    #create a form for the user to input data 
    st.subheader("Review Forum")
    benefits_review = st.text_area("benefits_review")
    side_effects_review = st.text_area("side_effects_review")
    comments_review = st.text_area("comments_review")

    if st.button("Predict Sentiment"):
        input_data = ["benefits_review", "side_effects_review", "comments_review"]
        prediction = model.predict(input_data)
        st.write("Model Predicts:", prediction[0])