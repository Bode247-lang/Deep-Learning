import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import streamlit as st
import speech_recognition as sr

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the text file and preprocess the data
with open('C:/Users/HP/Desktop/GOMYCODE/Chatbot/health.txt', 'r', encoding='utf-8') as f:
    data = f.read().replace('\n', ' ')
# Tokenize the text into sentences
sentences = sent_tokenize(data)
# Define a function to preprocess each sentence
def preprocess(sentence):
    # Tokenize the sentence into words
    words = word_tokenize(sentence)
    # Remove stopwords and punctuation
    words = [word.lower() for word in words if word.lower() not in stopwords.words('english') and word not in string.punctuation]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

# Preprocess each sentence in the text
corpus = [preprocess(sentence) for sentence in sentences]

# Defining the Similarity Function:
def get_most_relevant_sentence(query):
    # Preprocess the query
    query = preprocess(query)
    # Compute the similarity between the query and each sentence in the text
    max_similarity = 0
    most_relevant_sentence = ""
    for sentence in corpus:
        similarity = len(set(query).intersection(sentence)) / float(len(set(query).union(sentence)))
        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_sentence = " ".join(sentence)
    return most_relevant_sentence

# The chatbot Function:
def chatbot(question):
    most_relevant_sentence = get_most_relevant_sentence(question)
    return most_relevant_sentence

# Function to transcribe speech into text using speech recognition
def transcribe_speech():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        st.info("Speak now...")
        audio_text = r.listen(source)

        try:
            text = r.recognize_google(audio_text)
            return text
        except sr.RequestError:
            return "Sorry, the service is unavailable."
        except sr.UnknownValueError:
            return "Sorry, speech not recognized."

# Function to handle chatbot response for both text and speech inputs
def chatbot_response(input_text):
    # Sample code to illustrate handling both text and speech input
    if isinstance(input_text, str):
        # Text input
        response = chatbot(input_text)
    else:
        # Speech input
        transcribed_text = transcribe_speech()
        response = chatbot(transcribed_text)
    
    return response

# Streamlit app to incorporate speech recognition
def main():
    st.title("Speech-enabled Chatbot")
    st.write("Hello! I'm a chatbot designed by Clifford. Ask me any health-related question.")

    input_mode = st.radio("Select Input Mode:", ("Text", "Speech"))

    if input_mode == "Text":
        user_input = st.text_input("User:")
        if st.button("Submit"):
            bot_response = chatbot_response(user_input)
            st.write("Chatbot:", bot_response)
    else:  # Speech input
        if st.button("Start Recording"):
            text_from_speech = transcribe_speech()
            st.write(f"Transcription: {text_from_speech}")

            bot_response = chatbot_response(text_from_speech)
            st.write("Chatbot Response:", bot_response)

if __name__ == "__main__":
    main()
