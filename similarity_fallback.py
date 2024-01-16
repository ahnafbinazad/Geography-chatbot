import pandas as pd
import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class SimilarityFallback:
    def __init__(self, qa_kb_file='qa_kb.csv'):
        try:
            # Initialize the SimilarityFallback class with a Question-Answer knowledge base file
            self.qa_kb = pd.read_csv(qa_kb_file, quoting=csv.QUOTE_NONE, encoding='utf-8')
        except FileNotFoundError:
            print(f"Error: Knowledge base file '{qa_kb_file}' not found.")
            self.qa_kb = pd.DataFrame(columns=['Question', 'Answer'])

        # Initialize a TF-IDF vectorizer to convert text data into numerical vectors
        self.vectorizer = TfidfVectorizer()

        # Set up a set of English stopwords to filter out common words
        self.stop_words = set(stopwords.words('english'))

        # Initialize a lemmatizer for further text processing
        self.lemmatizer = WordNetLemmatizer()

        # Convert the Question-Answer knowledge base into a TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.qa_kb['Question'].apply(self.preprocess_text))

    def preprocess_text(self, text):
        # Preprocess the input text by converting to lowercase, removing stopwords, and lemmatizing
        tokens = [self.lemmatizer.lemmatize(word) for word in text.lower().split() if word not in self.stop_words]
        return ' '.join(tokens)

    def calculate_cosine_similarity(self, user_input):
        # Preprocess the user input
        processed_user_input = self.preprocess_text(user_input)

        # Convert the preprocessed user input into a TF-IDF vector
        user_tfidf = self.vectorizer.transform([processed_user_input])

        # Calculate cosine similarities between the user input vector and the knowledge base matrix
        cosine_similarities = cosine_similarity(user_tfidf, self.tfidf_matrix).flatten()

        # Find the index of the most similar question in the knowledge base
        most_similar_index = cosine_similarities.argmax()

        # Check if the similarity meets a certain threshold before considering it a match
        if cosine_similarities[most_similar_index] > 0.5:
            return most_similar_index
        else:
            return None

    def get_relevant_answer(self, most_similar_index):
        if most_similar_index is not None:
            # Retrieve and return the answer corresponding to the most similar question
            return self.qa_kb['Answer'].iloc[most_similar_index]
        else:
            return "I'm sorry, I couldn't find a relevant answer."
