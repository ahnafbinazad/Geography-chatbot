import pandas as pd
import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class SimilarityFallback:
    def __init__(self, qa_kb_file='qa_kb.csv'):
        """
        Initializes the SimilarityFallback class with a Question-Answer knowledge base file.

        Parameters:
        - qa_kb_file: String, path to the Question-Answer knowledge base file (default: 'qa_kb.csv').
        """
        try:
            self.qa_kb = pd.read_csv(qa_kb_file, quoting=csv.QUOTE_NONE, encoding='utf-8')
        except FileNotFoundError:
            print(f"Error: Knowledge base file '{qa_kb_file}' not found.")
            self.qa_kb = pd.DataFrame(columns=['Question', 'Answer'])

        self.vectorizer = TfidfVectorizer()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.qa_kb['Question'].apply(self.preprocess_text))

    def preprocess_text(self, text):
        """
        Preprocesses the input text by converting to lowercase, removing stopwords, and lemmatizing.

        Parameters:
        - text: String, input text to be preprocessed.

        Returns:
        - String: Preprocessed text.
        """
        tokens = [self.lemmatizer.lemmatize(word) for word in text.lower().split() if word not in self.stop_words]
        return ' '.join(tokens)

    def calculate_cosine_similarity(self, user_input):
        """
        Calculates the cosine similarity between the user input and questions in the knowledge base.

        Parameters:
        - user_input: String, user input text.

        Returns:
        - Integer or None: Index of the most similar question or None if no relevant match found.
        """
        processed_user_input = self.preprocess_text(user_input)
        user_tfidf = self.vectorizer.transform([processed_user_input])
        cosine_similarities = cosine_similarity(user_tfidf, self.tfidf_matrix).flatten()
        most_similar_index = cosine_similarities.argmax()
        if cosine_similarities[most_similar_index] > 0.5:
            return most_similar_index
        else:
            return None

    def get_relevant_answer(self, most_similar_index):
        """
        Retrieves the relevant answer based on the index of the most similar question.

        Parameters:
        - most_similar_index: Integer, index of the most similar question.

        Returns:
        - String: Relevant answer or a message indicating no relevant answer found.
        """
        if most_similar_index is not None:
            return self.qa_kb['Answer'].iloc[most_similar_index]
        else:
            return "I'm sorry, I couldn't find a relevant answer."
