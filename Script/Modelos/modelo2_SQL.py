import pandas as pd
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Establecer la conexión al motor de la base de datos Pos tgreSQL
engine = create_engine(f'postgresql://postgres:1234@127.0.0.1:5432/tfm')

# Consulta SQL para seleccionar solo las filas donde la columna "description" no sea nula
query = 'SELECT * FROM databooks WHERE "Description" IS NOT NULL;'

# Cargar los datos de la base de datos en un DataFrame de Pandas
databooks = pd.read_sql_query(query, con = engine)

# Preprocesamiento de datos (opcional dependiendo de la limpieza previa del dataset)

# Construir la matriz TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2, max_df=0.5)
tfidf_matrix = tfidf_vectorizer.fit_transform(databooks['Description'])


# Función para recomendar libros
def recommend_books(user_preference, n=5):
    # Vectorizar la preferencia del usuario
    user_preference_vector = tfidf_vectorizer.transform([user_preference])

    # Calcular la similitud del coseno entre la preferencia del usuario y las descripciones de los libros
    cosine_similarities = cosine_similarity(user_preference_vector, tfidf_matrix).flatten()

    # Obtener los índices de los libros más similares
    most_similar_indices = cosine_similarities.argsort()[-n:][::-1]

    # Devolver los libros recomendados
    recommended_books = databooks.iloc[most_similar_indices]
    return recommended_books


# Ejemplo de interacción con el usuario
user_preference = input("Ingrese sus preferencias de lectura: ")
recommended_books = recommend_books(user_preference)
print("Libros recomendados:")
print(recommended_books[['Name', 'Authors', 'Description']])
