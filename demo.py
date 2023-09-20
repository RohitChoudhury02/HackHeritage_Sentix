import streamlit as st
import requests
from wordcloud import WordCloud  # Import WordCloud here if it's not already imported
import matplotlib.pyplot as plt
import pandas as pd
# Streamlit UI
st.title("Sentiment Analysis with FastAPI and Keras")

# Text input box for user input
user_input = st.text_area("Enter text:", "")

# Button to trigger the prediction
if st.button("Predict Sentiment"):
    if user_input:
        # Show a spinner while waiting for the response
        with st.spinner("Predicting sentiment..."):
            # Define the FastAPI endpoint URL
            endpoint_url = "http://localhost:8000/predict"  # Update with your FastAPI server's URL

            # Send a GET request to the FastAPI endpoint with the user's input
            response = requests.get(endpoint_url, params={"link": user_input})

            if response.status_code == 200:
                result = response.json()

                # Display the sentiment prediction and probability with emojis
                st.subheader("Sentiment Prediction:")
                sentiment = result['sentiment']
                prediction = result['prediction']
                emoji = "😃" if sentiment == "positive" else "😞" if sentiment == "negative" else "😐"
                st.markdown(f"**Text:** {result['text']}")
                st.markdown(f"**Sentiment:** {sentiment} {emoji}")
                st.markdown(f"**Probability:** {prediction:.6f}")

                # Generate and display the Word Cloud
                st.subheader("Word Cloud of Tweet Text:")
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(result['text'])
                plt.figure(figsize=(11, 10))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.title("Word Cloud of Tweet Text")
                plt.axis('off')
                st.pyplot(plt)
                # Sentiment Distribution Bar Chart
                st.subheader("Sentiment Distribution:")
                sentiment_data = pd.DataFrame({'Sentiment': ['Positive', 'Negative', 'Neutral'],'Count': [result['positive_count'], result['negative_count'], result['neutral_count']]})
                st.bar_chart(sentiment_data.set_index('Sentiment'))

                # Word Frequency Analysis Bar Chart (Top 10 words)
                st.subheader("Top 10 Most Frequent Words:")
                word_frequency_data = pd.DataFrame(result['word_frequency'], columns=['Word', 'Frequency'])
                st.bar_chart(word_frequency_data.head(10).set_index('Word'))

            else:
                st.error(f"Error: {response.status_code} - Unable to get sentiment prediction.")
    else:
        st.warning("Please enter some text for sentiment prediction.")
