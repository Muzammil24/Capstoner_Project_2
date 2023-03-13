# Core Pkgs
import streamlit as st
import altair as alt
import plotly.express as px
from nrclex import NRCLex

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# EDA Pkgs
import pandas as pd
import numpy as np
from datetime import datetime


# Utils
import joblib

pipe_lr = joblib.load(
    open("emotion_classifier_pipe_lr_03_june_2021.pkl", "rb"))


# Fxn 1

def convert_to_df(sentiment):
    sentiment_dict = {'polarity': sentiment.polarity,
                      'subjectivity': sentiment.subjectivity}
    sentiment_df = pd.DataFrame(
        sentiment_dict.items(), columns=['metrics', 'value'])
    return sentiment_df


def analyze_token_sentiment(docx):
    analyzer = SentimentIntensityAnalyzer()
    pos_list = []
    neg_list = []
    neu_list = []

    for i in docx.split():
        res = analyzer.polarity_scores(i)['compound']
        if res >= 0.1:
            pos_list.append(i)
            pos_list.append(res)

        elif res <= -0.1:
            neg_list.append(i)
            neg_list.append(res)

        else:
            neu_list.append(i)

    result = {'positive': pos_list, 'negative': neg_list, 'neutral': neu_list}

    return result


# Fxn

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗",
                       "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"}

# Main Application


def main():

    st.title("Emotion Classifier App")
    menu = ["About", "MultinomialNB", "TextBlob", "NRCLex"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "MultinomialNB":
        # add_page_visited_details("Home", datetime.now())
        st.subheader("Emotion In Text")

        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1, col2 = st.columns(2)

            # Apply Fxn Here
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)
            # add_prediction_details(raw_text, prediction,
            #                        np.max(probability), datetime.now())

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction, emoji_icon))
                st.write("Confidence:{}".format(np.max(probability)))

            with col2:
                st.success("Prediction Probability")
                # st.write(probability)
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                # st.write(proba_df.T)

                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(
                    x='emotions', y='probability', color='emotions')
                st.altair_chart(fig, use_container_width=True)

    elif choice == "TextBlob":
        st.subheader("TextBlob")
        with st.form(key='nlpForm'):
            raw_text = st.text_area("Enter Text Here")
            submit_button = st.form_submit_button(label="Analyze")

        # layout

        col1 = st.columns(1)

        if submit_button:

            st.info("Results")
            sentiment = TextBlob(raw_text).sentiment
            # st.write(sentiment)

            #  Emoji

            if sentiment.polarity > 0:
                st.markdown("Sentiment:: Positive :🤗 ")

            elif sentiment.polarity < 0:
                st.markdown("Sentiment:: Negative :😠 ")

            else:
                st.markdown("Sentiment:: Neutral :😐 ")

            # DataFrame

            results_df = convert_to_df(sentiment)
            st.dataframe(results_df)

            # Visualization
            c = alt.Chart(results_df).mark_bar().encode(
                x='metrics',
                y='value', color='metrics')
            st.altair_chart(c, use_container_width=True)

    elif choice == "NRCLex":
        st.subheader("NRCLex")

        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

            col1 = st.columns(1)

            if submit_text:
                emotion = NRCLex(raw_text)
                j = emotion.affect_frequencies

                st.success("Bar Chart")
                # st.write(j)

                data_list = [{'Emotions': k, 'value': v} for k, v in j.items()]

                st.bar_chart(data_list, x='Emotions', y='value')

    else:

        st.subheader("NRC LEXICON")
        st.write("""      
            The NRC lexicon is a lexicon-based approach to sentiment analysis that was developed by the National Research Council of Canada. The lexicon contains over 140,000 words and phrases that are associated with specific emotions and sentiments, such as joy, trust, anger, fear, and sadness.

            Each word and phrase in the lexicon is assigned a score or weight for each of the eight basic emotions, as well as for positive and negative sentiment. The scores are based on the results of a crowdsourcing experiment in which participants were asked to rate the emotional content of a large set of words and phrases.
            
            NRC lexicon is a powerful tool for sentiment analysis that can provide accurate and consistent sentiment scores for a wide range of emotions. It is a valuable resource for businesses that need to analyze sentiment in text data.
         """)

        st.subheader("TEXTBLOB")
        st.write("""      
            TextBlob is a Python library that provides a simple and intuitive interface for performing common natural language processing (NLP) tasks, including sentiment analysis. It is built on top of the popular NLTK library and provides an easy-to-use API for processing text data.

            The TextBlob class in TextBlob represents a text document and provides various methods for performing NLP tasks, including sentiment analysis. The sentiment property of a TextBlob object returns a tuple of two values: the polarity and subjectivity of the text.

            The polarity score indicates the sentiment of the text, ranging from -1 (very negative) to 1 (very positive), with 0 indicating a neutral sentiment. The subjectivity score indicates the degree of subjective vs objective language used in the text, ranging from 0 (very objective) to 1 (very subjective).

            Here's an example program that demonstrates how to use TextBlob to perform sentiment analysis on a piece of text:
         """)


if __name__ == '__main__':
    main()
