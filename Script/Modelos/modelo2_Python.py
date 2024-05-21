
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def train_model(path_books):
    df_books = pd.read_csv(path_books)
    databooks = df_books[df_books['Description'].notnull()]

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2, max_df=0.5)
    tfidf_matrix = tfidf_vectorizer.fit_transform(databooks['Description'])
    
    return tfidf_vectorizer, tfidf_matrix, databooks

def save_model(tfidf_vectorizer, tfidf_matrix, databooks, path_model):
    with open(path_model, 'wb') as f:
        pickle.dump((tfidf_vectorizer, tfidf_matrix, databooks), f)

def load_model(path_model):
    with open(path_model, 'rb') as f:
        return pickle.load(f)

def recommend_books(user_preference, tfidf_vectorizer, tfidf_matrix, databooks, n=5):
    user_preference_vector = tfidf_vectorizer.transform([user_preference])

    cosine_similarities = cosine_similarity(user_preference_vector, tfidf_matrix).flatten()

    most_similar_indices = cosine_similarities.argsort()[-n:][::-1]

    recommended_books = databooks.iloc[most_similar_indices]
    
    return recommended_books
