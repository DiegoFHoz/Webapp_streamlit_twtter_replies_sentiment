import streamlit as st
import tensorflow as ts
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from transformers import pipeline


#print(classifier('una mierda'))

def main():
    classifier = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')

    replies_df = (
        "/home/dfh/Descargas/bank_cleanedV1.csv"
    )

    st.title("Sentiment Analysis of Tweets about US Airlines")
    st.sidebar.title("Sentiment Analysis of Tweets")
    st.markdown("This application is a Streamlit dashboard used "
            "to analyze sentiments of tweets 🐦")
    st.sidebar.markdown("This application is a Streamlit dashboard used "
            "to analyze sentiments of tweets 🐦")


    tweet = st.text_input("Enter Your Tweet","Type Here..")
    score=classifier(tweet)
    #[0].get('label')

    st.write(score)

if __name__ == "__main__":
    main()