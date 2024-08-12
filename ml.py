# Importaciones de sklearn para el modelo de recomendacion.
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.sparse import hstack

from geopy.distance import geodesic

# importar sqlalchemy.
from sqlalchemy import create_engine

# importar pandas.
import pandas as pd

# Import desde models.
from models import engine

#engine = create_engine('sqlite:///data/movies.db')

# Cargar datos a Dataframe.
data = pd.read_sql('restaurants', engine)

columnas_a_procesar = ['name', 'address',  'combined_categories']

# Vectorizar cada columna procesada usando TF-IDF.
dic_vectorizadores = {}
list_matrices = []

# Vectorizar los datos con Tfid.
for column in columnas_a_procesar:
    vectorizer = TfidfVectorizer(max_features=31000)
    matrizado = vectorizer.fit_transform(data[f'{column}'])
    dic_vectorizadores[column] = vectorizer
    list_matrices.append(matrizado)

# Combinar todas las matrices TF-IDF en una sola matriz si es necesario.
combinacion_matrices = hstack(list_matrices).tocsr() if len(list_matrices) > 1 else list_matrices[0]

# Calcular la similitud del coseno bajo demanda.
def calcular_similitud_coseno(indice_x, matrix):
    return cosine_similarity(matrix[indice_x], matrix).flatten()

# Calcular la distancia geográfica entre dos ubicaciones
def calcular_distancia(coord1, coord2):
    return geodesic(coord1, coord2).kilometers


def get_recommendations(name, data, top_n=5, max_dist_km=10):
    if name not in data['name'].values:
        return f"El restaurante '{name}' no se encuentra en nuestra base de datos."
    
    indice_x = data[data['name'] == name].index[0]
    coord_restaurante = (data['latitude'].iloc[indice_x], data['longitude'].iloc[indice_x])
    resultados = calcular_similitud_coseno(indice_x, combinacion_matrices)
    
    resultados = list(enumerate(resultados))
    resultados = sorted(resultados, key=lambda x: x[1], reverse=True)
    
    nombres_unicos = set()
    recomendaciones = []
    
    for idx, score in resultados[1:]:
        nombre = data['name'].iloc[idx]
        avg_rating = data['avg_rating'].iloc[idx]
        direccion = data['address'].iloc[idx].replace(f'{nombre}, ', '')
        coord_recomendacion = (data['latitude'].iloc[idx], data['longitude'].iloc[idx])
        distancia = calcular_distancia(coord_restaurante, coord_recomendacion)
        
        if nombre not in nombres_unicos and distancia <= max_dist_km:
            recomendaciones.append((nombre, direccion, avg_rating, distancia))
            nombres_unicos.add(nombre)
        
        if len(recomendaciones) == top_n:
            break

    # Ordenar las recomendaciones por avg_rating en orden descendente
    recomendaciones_ordenadas = sorted(recomendaciones, key=lambda x: x[2], reverse=True)
    
    return [(rec[0], rec[1], rec[2]) for rec in recomendaciones_ordenadas]
 