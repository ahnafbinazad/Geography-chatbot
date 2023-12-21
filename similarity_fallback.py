import pandas as pd
import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords


class SimilarityFallback:
    def __init__(self, qa_kb_file='qa_kb.csv'):
        # Initialize the SimilarityFallback class with a Question-Answer knowledge base file
        self.qa_kb = pd.read_csv(qa_kb_file, quoting=csv.QUOTE_NONE, encoding='utf-8')

        # Initialize a TF-IDF vectorizer to convert text data into numerical vectors
        self.vectorizer = TfidfVectorizer()

        # Set up a set of English stopwords to filter out common words
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        # Preprocess the input text by converting to lowercase and removing stopwords
        tokens = [word for word in text.lower().split() if word not in self.stop_words]
        return ' '.join(tokens)

    def calculate_cosine_similarity(self, user_input):
        # Preprocess the user input
        processed_user_input = self.preprocess_text(user_input)

        # Convert the Question-Answer knowledge base into a TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(self.qa_kb['Question'].values)

        # Convert the preprocessed user input into a TF-IDF vector
        user_tfidf = self.vectorizer.transform([processed_user_input])

        # Calculate cosine similarities between the user input vector and the knowledge base matrix
        cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()

        # Find the index of the most similar question in the knowledge base
        most_similar_index = cosine_similarities.argmax()
        return most_similar_index

    def get_relevant_answer(self, most_similar_index):
        # Retrieve and return the answer corresponding to the most similar question
        return self.qa_kb['Answer'].iloc[most_similar_index]
