import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Funciones -------------------------------------------------------------------------
## Rating Numérico

def map_ratings_to_numeric(dataframe, rating_column):
    rating_mapping = {
        'it was amazing': 5,
        'really liked it': 4,
        'liked it': 3,
        'it was ok': 2,
        'did not like it': 1,
        "This user doesn't have any rating": 0
    }

    dataframe['NumericRating'] = dataframe[rating_column].map(rating_mapping)
    
    return dataframe

## Ratings
def process_ratings_files(df_usuarios):
    # Realizar la transformación de las columnas 'Rating' a valores numéricos
    map_ratings_to_numeric(df_usuarios, 'Rating')
    ratings = df_usuarios.rename(columns={'ID':'IdUser'})
    return ratings


#2. Modelo 1: Recomendaciones por usuario -------------------------------------------------------------------------
def modelo1 (path_ratings, path_recomendations):
    """ Entra como parámetro el path de la tabla de ratings y el path para guardar recomendaciones del modelo 1.
        No retorna nada.
        Guard en el en segundo path una tabla con Los 3 liked books y las tres recomendaciones """
        
    df_usuarios = pd.read_csv(path_ratings)
    ratingsR = process_ratings_files(df_usuarios)


    df = ratingsR.copy()
    # Ordenamos el DataFrame por Usuario ID y luego por rating de forma descendente
    df = df[(df['NumericRating'] != 0)]
    df = df.sort_values(['IdUser','NumericRating'], ascending=[True, False])

    # Filtramos solo los libros que les gustaron a los usuarios
    liked_books = df[df['NumericRating'] > 2]

    liked_books_sorted = liked_books.sort_values(by=['IdUser', 'NumericRating'], ascending=[True, False])

    # Creamos un DataFrame que contiene, para cada usuario, una lista de los libros que les gustaron

    user_books = liked_books_sorted.groupby('IdUser')['BookName'].apply(list).reset_index(name='LikedBooks')

    # Creamos un diccionario para almacenar las recomendaciones para cada usuario
    recommendations = {}

    # Creamos un vectorizador para la matriz de similitud entre usuarios basada en los libros que les gustaron
    vectorizer = TfidfVectorizer(stop_words='english')
    user_matrix = vectorizer.fit_transform(user_books['LikedBooks'].apply(lambda x: ' '.join(x)))

    # Calculamos la similitud entre usuarios usando la similitud del coseno
    user_similarity = cosine_similarity(user_matrix, user_matrix)

    # Iteramos sobre cada usuario y sus libros para hacer recomendaciones
    for index, user_row in user_books.iterrows():
        user_id = user_row['IdUser']
        user_liked_books = user_row['LikedBooks']

        # Almacenamos las recomendaciones en el diccionario
        if len(user_liked_books) >= 2:
            # Obtenemos los usuarios similares basándonos en los libros que les gustaron
            similar_users_indices = user_similarity[index].argsort()[::-1][1:4]

            # Filtramos los libros que ya le gustaron al usuario
            user_read_books = liked_books[liked_books['IdUser'] == user_id]['BookName'].tolist()
            similar_books = user_books.iloc[similar_users_indices]['LikedBooks'].explode().unique()
            similar_books = [book for book in similar_books if book not in user_read_books][:3]
            
            # Sufficient liked books to make recommendations
            recommendations[user_id] = {
                'liked_books': user_liked_books,
                'recommended_books': similar_books
            }
        else:
            # Not enough liked books to make recommendations
            recommendations[user_id] = {
                'liked_books': user_liked_books,
                'recommended_books': ["No hay datos suficientes para realizar una recomendación"] }
            
    data = []

    # Iterate over the recommendations dictionary
    for user_id, books in recommendations.items():
        # Extract liked books and recommended books
        liked_books = books['liked_books'][:3]  # Slice to include only the first three liked books
        recommended_books = books['recommended_books']
        
        # Create a row for each user with liked and recommended books
        row = {'user_id': user_id}
        
        # Add liked books to the row
        for i, book in enumerate(liked_books, start=1):
            row[f'liked_book_{i}'] = book
        
        # Add recommended books to the row
        for i, book in enumerate(recommended_books[:3], start=1):
            row[f'recommended_book_{i}'] = book
        
        # Append the row to the data list
        data.append(row)

    # Convert the data list to a DataFrame
    recomendation_model1 = pd.DataFrame(data)

    recomendation_model1['recommended_book_1'] = np.where(recomendation_model1['recommended_book_1'].isnull(), "No hay datos suficientes para realizar una recomendación", recomendation_model1['recommended_book_1'])
    recomendation_model1.to_csv(path_recomendations, index=False)

    return None


#3. Recomendaciones desde csv ---------------------------------------------------------------------------------------

def recomendacion_modelo1 (path_recomendations, Usuario):
    
    recomendation_model1 = pd.read_csv(path_recomendations)
    
    recomendacion1_usuario =recomendation_model1.loc[recomendation_model1['user_id']==Usuario, 'recommended_book_1'].values[0]
    recomendacion2_usuario =recomendation_model1.loc[recomendation_model1['user_id']==Usuario, 'recommended_book_2'].values[0]
    recomendacion3_usuario =recomendation_model1.loc[recomendation_model1['user_id']==Usuario, 'recommended_book_3'].values[0]

    return print(f'Las recomendaciones para el usuario número {Usuario} son: \n 1. {recomendacion1_usuario} \n 2. {recomendacion2_usuario} \n 3. {recomendacion3_usuario}')