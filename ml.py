# Importaciones de sklearn para el modelo de recomendacion.
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.sparse import hstack

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


from geopy.distance import geodesic

# importar sqlalchemy.
from sqlalchemy import create_engine

# importar pandas.
import pandas as pd

# Import desde models.
from models import engine

engine = create_engine('sqlite:///data/restaurants.db')

# Cargar datos a Dataframe.
data = pd.read_sql('restaurants', engine)

# Modelo de prediccion
data_rec = data[data['COUNTY_NAM']== 'los angeles']
columnas_a_procesar = ['name', 'address', 'cluster_categories']

# Vectorizar cada columna procesada usando TF-IDF.
dic_vectorizadores = {}
list_matrices = []

# Vectorizar los datos con Tfid.
for column in columnas_a_procesar:
    vectorizer = TfidfVectorizer(max_features=31000)
    matrizado = vectorizer.fit_transform(data_rec[f'{column}'])
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

def get_recommendations(name, data_rec, top_n=5, max_dist_km=10):

    name = name.lower()
    if name not in data_rec['name'].values:
        return f"El restaurante '{name}' no se encuentra en nuestra base de datos."
    
    indice_x = data_rec[data_rec['name'] == name].index[0]
    coord_restaurante = (data_rec['latitude'].iloc[indice_x], data_rec['longitude'].iloc[indice_x])
    resultados = calcular_similitud_coseno(indice_x, combinacion_matrices)
    
    resultados = list(enumerate(resultados))
    resultados = sorted(resultados, key=lambda x: x[1], reverse=True)
    
    nombres_unicos = set()
    recomendaciones = []
    
    for idx, score in resultados[1:]:
        nombre = data_rec['name'].iloc[idx]
        avg_rating = data_rec['avg_rating'].iloc[idx]
        direccion = data_rec['address'].iloc[idx].replace(f'{nombre}, ', '')
        coord_recomendacion = (data_rec['latitude'].iloc[idx], data_rec['longitude'].iloc[idx])
        distancia = calcular_distancia(coord_restaurante, coord_recomendacion)
        caracteristicas_clave = data_rec['caracteristicas_clave'].iloc[idx]
        
        if nombre not in nombres_unicos and distancia <= max_dist_km:
            recomendaciones.append((nombre, direccion, avg_rating, distancia, caracteristicas_clave))
            nombres_unicos.add(nombre)
        
        if len(recomendaciones) == top_n:
            break

    # Ordenar las recomendaciones por avg_rating en orden descendente
    recomendaciones_ordenadas = sorted(recomendaciones, key=lambda x: x[2], reverse=True)
    
    # Retornar las recomendaciones incluyendo la columna 'caracteristicas_clave'
    return [(rec[0], rec[1], rec[2], rec[4]) for rec in recomendaciones_ordenadas]

# --------------------------------------------------------------------------------------------------------------------------------

# Modelo de prediccion...

# Agrupar por categoria y id de la categoria

filtrado = data[data['reviews_total']!=0]

#Dividir data set
features=['county_id', 'cluster', 'negative_sentiment','positive_sentiment']
X = filtrado[features]
y = filtrado['success']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)




