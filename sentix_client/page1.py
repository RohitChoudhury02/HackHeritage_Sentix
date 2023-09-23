import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
from bs4 import BeautifulSoup
import requests
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot
# Define label mapping for emotions
label_mapping = {0: 'anger', 1: 'dread', 2: 'joy', 3: 'sadness', 4: 'neutral', 5: 'surprise', 6: 'shame', 7: 'disgust'}

# Load your sentiment analysis model and tokenizer here
with open('../models/model.pkl', 'rb') as tokenizer_file:
    loaded_model = pickle.load(tokenizer_file)
with open('../models/tokenizer_sih.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Load your emotion analysis model here
# model = load_model('../models/emix.model.h5')

# Define custom functions for text cleaning and sentiment categorization
def clean_text(tweet):
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet)
    tweet = re.sub(r"@\w+|#\w+", "", tweet)
    tweet = re.sub(r"[^a-zA-Z\s]", "", tweet)
    tweet = tweet.lower()
    tweet = re.sub(r"\s+", " ", tweet).strip()
    return tweet

def preprocess_data(data):
    vocab_size = 5000  # Should match the vocab size used during model training
    onehot_repr = [one_hot(words, vocab_size) for words in [data]]
    sent_length = 100  # Should match the sequence length used during model training
    docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)
    return docs

def categorize_sentiment(predictions):
    thresholds = [(0.97, "Highly Positive"), (0.8, "Positive"), (0.3, "Neutral"), (0.2, "Negative")]
    for threshold, label in thresholds:
        if predictions >= threshold:
            return label
    return "Very Negative"

# Function to perform emotion analysis (you can use a more advanced model here)
def perform_emotion_analysis(text):
    # preprocessed_text = preprocess_data(text)
    # e_prediction = model.predict(preprocessed_text)
    # predicted_label = label_mapping[np.argmax(e_prediction)]
    return 0

# Streamlit app
def app():
    
   
    # Define custom styles
    button_style = (
        "background-color: #4CAF50; color: white; border-radius: 5px;"
        "padding: 10px 20px; font-size: 16px; border: none; cursor: pointer;"
    )

    loader_style = (
        "color: #4CAF50; display: inline-block; border: 4px solid #f3f3f3; "
        "border-top: 4px solid #3498db; border-radius: 50%; width: 20px; height: 20px; "
        "animation: spin 2s linear infinite; margin-right: 10px;"
    )

    # Add custom CSS
    st.markdown(
        f'<style>.stButton > button{{ {button_style} }}</style>',
        unsafe_allow_html=True,
    )
    
    st.markdown(
        f'<style>@keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}</style>',
        unsafe_allow_html=True,
    )
    st.title("Social Media Sentiment Analysis")

    # Create tabs for different social media platforms
    selected_tab = st.selectbox("Select a Social Media Platform", ["YouTube 🔥", "Instagram ✅", "LinkedIn 💥"])

    # Input box for the user to enter a link
    link = st.text_input(f"Enter a {selected_tab} Link:")

    if st.button("Analyze Sentiment"):
        if link:
            with st.spinner("Fetching and analyzing data..."):
                try:
                    # Scrape data from the provided link
                    response = requests.get(link)
                    soup = BeautifulSoup(response.content, "html.parser")
                    if selected_tab == "YouTube":
                        # Extract text from YouTube video description
                        description_element = soup.find('meta', {'name': 'description'})
                        if description_element:
                            video_description = description_element['content']
                            cleaned_text = clean_text(video_description)
                        # Perform sentiment analysis
                            text_sequences = tokenizer.texts_to_sequences([cleaned_text])
                            text_sequences = pad_sequences(text_sequences, maxlen=100)
                            predictions = loaded_model.predict(text_sequences)
                            sentiment = categorize_sentiment(predictions[0][0])
                            
                            # Perform emotion analysis
                            # emotion = perform_emotion_analysis(cleaned_text)

                            # Display results
                            st.subheader("Sentiment:")
                            st.write(sentiment)

                            st.subheader("Emotion:")
                            # st.write(emotion)
                    elif selected_tab == "Instagram":
                        # Extract text from Instagram post (you may need to adapt this for Instagram)
                        text_data = soup.get_text()
                        cleaned_text = clean_text(text_data)
                        # Perform sentiment analysis
                        text_sequences = tokenizer.texts_to_sequences([cleaned_text])
                        text_sequences = pad_sequences(text_sequences, maxlen=100)
                        predictions = loaded_model.predict(text_sequences)
                        sentiment = categorize_sentiment(predictions[0][0])
                        
                        # Perform emotion analysis
                        # emotion = perform_emotion_analysis(cleaned_text)

                        # Display results
                        st.subheader("Sentiment:")
                        st.write(sentiment)

                        st.subheader("Emotion:")
                        # st.write(emotion)
                    elif selected_tab == "LinkedIn":
                        # Extract text from LinkedIn post (you may need to adapt this for LinkedIn)
                        text_data = soup.get_text()
                        start_index = text_data.find("Report this post")
                        end_index = text_data.find("Like")
                        substring = text_data[start_index + 51:end_index]

                        cleaned_text = re.sub(r'\s+', ' ', substring)
                        # Perform sentiment analysis
                        text_sequences = tokenizer.texts_to_sequences([cleaned_text])
                        text_sequences = pad_sequences(text_sequences, maxlen=100)
                        predictions = loaded_model.predict(text_sequences)
                        sentiment = categorize_sentiment(predictions[0][0])
                        
                        # Perform emotion analysis
                        # emotion = perform_emotion_analysis(cleaned_text)

                        # Display results
                        st.subheader("Sentiment:")
                        st.write(sentiment)

                        st.subheader("Emotion:")
                        # st.write(emotion)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                        
                    
        else:
            st.warning("Please enter a link for scraping and sentiment analysis.")

if __name__ == "__main__":
    app()
