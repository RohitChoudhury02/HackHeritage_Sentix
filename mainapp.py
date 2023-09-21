
import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
from bs4 import BeautifulSoup
import requests
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

PAGE_HOME = "Home"
PAGE_PAGE1 = "Page 1"
PAGE_PAGE2 = "Page 2"
PAGE_PAGE3 = "Page 3"

# Create a sidebar with navigation links
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", [PAGE_HOME, PAGE_PAGE1, PAGE_PAGE2, PAGE_PAGE3],key="page_selector")

def home_page():
    st.title("This is Sentix")
    st.write("Your go to sentiment analysis tool!")

    st.title("Analyze sentiment using link of different social media post")
    page=st.sidebar.selectbox("Choose a social media platform", ["Instagram","Facebook","Linkedin"],key="platform_selector")
    def page4():
        plt.title("instagram")
    # Load your LSTM model, tokenizer, and other necessary data
        with open('model.pkl', 'rb') as model_file:
            loaded_model = pickle.load(model_file)

        with open('tokenizer_sih.pkl', 'rb') as tokenizer_file:
            tokenizer = pickle.load(tokenizer_file)

        # Define custom functions for text cleaning and sentiment categorization
        def clean_text(tweet):
            # Add your custom text cleaning logic here
            tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet)
            tweet = re.sub(r"@\w+|#\w+", "", tweet)
            tweet = re.sub(r"[^a-zA-Z\s]", "", tweet)
            tweet = tweet.lower()
            tweet = re.sub(r"\s+", " ", tweet).strip()
            return tweet

        def categorize_sentiment(predictions):
            thresholds = [(0.97, "Highly Positive"), (0.8, "Positive"), (0.3, "Neutral"), (0.2, "Negative")]
            for threshold, label in thresholds:
                if predictions >= threshold:
                    return label
            return "Very Negative"

        def generate_wordcloud(text_data):
            # Join the text data into a single string
            text = ' '.join(text_data)
            
            # Generate the WordCloud
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

            # Display the WordCloud using Matplotlib
            plt.figure(figsize=(11, 10))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title("Word Cloud of Tweet Text")
            plt.axis('off')
            return plt

        # Streamlit UI
        

        # Text input box for user to enter a link
        link_input = st.text_input("Enter a link to scrape data and analyze sentiment:")

        # Button to trigger scraping and sentiment analysis
        if st.button("Scrape and Analyze Sentiment"):
            if link_input:
                try:
                    # Scrape data from the provided link
                    response = requests.get(link_input)
                    soup = BeautifulSoup(response.content, "html.parser")
                    text_data = soup.get_text()

                    # Clean the scraped text
                    cleaned_text = clean_text(text_data)

                    # Tokenize and pad the text data
                    text_sequences = tokenizer.texts_to_sequences([cleaned_text])
                    text_sequences = pad_sequences(text_sequences, maxlen=100)

                    # Make predictions using the loaded model
                    predictions = loaded_model.predict(text_sequences)

                    # Categorize sentiment
                    sentiment = categorize_sentiment(predictions[0][0])

                    # Display sentiment analysis results
                    st.subheader("Sentiment Analysis Results:")
                    st.markdown(f"**Scraped Text:**\n{cleaned_text}")
                    st.markdown(f"**Sentiment:** {sentiment}")
                    
                    # Generate and display the Word Cloud
                    st.subheader("Word Cloud of Scraped Text:")
                    wordcloud = generate_wordcloud(cleaned_text)
                    st.pyplot(wordcloud)
                    
                    # Debugging: Print the values for troubleshooting
                    print("Scraped Text:", text_data)
                    print("Sentiment:", sentiment)

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    print("Error:", str(e))
            else:
                st.warning("Please enter a link for scraping and sentiment analysis.")
    def page5():
        plt.title("facebook")
    def page6():
        plt.title("linkedin")
    if page == "Instagram":
        page4()
    elif page == "Facebook":
        page5()
    elif page == "Linkedin":
        page6()



def page2():
    tab1, tab2 = st.tabs(["Dataset", "Model"])
    tab1.subheader("Characteristics of Dataset")
    tab3,tab4,tab7=tab1.tabs(["Labels","Varied sizes","word cloud"])
    tab3.subheader("distribution of sentiments")
    tab3.image("./image/bar.png")
    tab3.image("./image/pie.png")
    tab4.subheader("Distribution showing the different lengths of tweets")
    tab4.image("./image/length.png")
    tab2.subheader("Model working")
    tab7.subheader("positive word cloud")
    tab7.image("./image/positive.png")
    tab7.subheader("negative word cloud")
    tab7.image("./image/negative.png")
    tab5,tab6=tab2.tabs(["Confusion Matrix","plots"])

def page3():
    st.title("Load a CSV file to predict output")
    with open('model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)

    # Load the tokenizer used during model training
    with open('tokenizer_sih.pkl', 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)

    def clean_text(text):
        cleaned_text = text.lower()
        cleaned_text = cleaned_text.replace('!', ' ')
        return cleaned_text

    st.sidebar.title("Settings")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.write(df.head())

        # Assuming your text data column is named 'Tweet'
        text_data = df['Tweet'].astype(str)

        # Tokenize and pad the text data
        text_sequences = tokenizer.texts_to_sequences(text_data)
        text_sequences = pad_sequences(text_sequences, maxlen=100)  # Assuming max_sequence_length is 100

        # Make predictions using the loaded model
        predictions = loaded_model.predict(text_sequences)

        # Add predictions to the DataFrame
        df['Predicted_Sentiment'] = predictions
        thresholds = [(0.97, "Highly Positive"), (0.8, "Positive"), (0.3, "Neutral"), (0.2, "Negative")]

        def categorize_sentiment(predictions):
            for threshold, label in thresholds:
                if predictions >= threshold:
                    return label
            return "Very Negative"

        df['Sentiment'] = predictions
        df['Sentiment'] = df['Sentiment'].apply(categorize_sentiment)
        st.write("Predictions and Sentiments:")
        st.write(df[['Tweet', 'Sentiment','Predicted_Sentiment']])
        if st.button("Download CSV File"):
            # Create a CSV file from the DataFrame
            csv_file = df.to_csv(index=False)

            # Offer the CSV file for download
            st.download_button(
                label="Download CSV",
                data=csv_file.encode(),
                file_name="sentiment_predictions.csv",
                key="download_button",
                mime="text/csv",
            )
        st.write("       ")

        # Clean the text data using the custom cleaner
        text_data = text_data.apply(clean_text)
        wordcloud = generate_wordcloud(text_data)
        st.pyplot(wordcloud)

def generate_wordcloud(text_data):
    # Join the text data into a single string
    text = ' '.join(text_data)

    # Generate the WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Display the WordCloud using Matplotlib
    plt.figure(figsize=(11, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title("Word Cloud of Tweet Text")
    plt.axis('off')
    return plt

if __name__ == "__main__":
    if page == PAGE_HOME:
        home_page()
    #elif page == PAGE_PAGE1:
     #   page1()
    elif page == PAGE_PAGE2:
        page2()
    elif page == PAGE_PAGE3:
        page3()

