import os
import json
import pandas as pd
from geopy.distance import great_circle
import folium
from folium.plugins import HeatMap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.cluster import KMeans
import nltk
from collections import Counter
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


import warnings
warnings.filterwarnings('ignore')
# class Negocios:
#     def __init__(self,) :
        
california_lat_min = 32.5343
california_lat_max = 42.0095
california_lon_min = -124.4096
california_lon_max = -114.1315

california_df=pd.DataFrame()
df_empty = pd.DataFrame()
df_combined=pd.DataFrame()

# Función para calcular la distancia desde Blythe
def is_within_range(row, location, threshold):
    point = (row['latitude'], row['longitude'])
    distance = great_circle(location, point).kilometers
    return distance <= threshold

def lectura_de_google():
    global california_df
    # Leer el archivo Parquet
    #df = pd.read_parquet('src/data-origen/google/google_metadata/google_metadata.parquet', engine='pyarrow')
    df = pd.read_parquet('Data-origen/google_metadata.parquet', engine='pyarrow')
    #df = pd.read_parquet('/opt/airflow/source-data/google_metadata.parquet', engine='pyarrow')
    
    df_meta_data=df.loc[:,['gmap_id','address','name', 'latitude', 'longitude','category','num_of_reviews','avg_rating']]
    # Filtra el DataFrame para que solo incluya registros dentro de los límites de California
    california_df = df_meta_data[(df_meta_data['latitude'] >= california_lat_min) & (df_meta_data['latitude'] <= california_lat_max) &
                   (df_meta_data['longitude'] >= california_lon_min) & (df_meta_data['longitude'] <= california_lon_max)]
    print(california_df.shape)
    return california_df

def limpieza_de_google():
    global california_df
    # Encontrar filas duplicadas basadas en la columna 'gmap_id'
    #duplicated_rows = california_df[california_df.duplicated(subset='gmap_id', keep=False)]
    # Eliminar duplicados basados en la columna 'gmap_id'
    california_df = california_df.drop_duplicates(subset='gmap_id')
    california_df = california_df.dropna(subset=['address'])
    california_df = california_df.dropna(subset=['category'])
    
    nevada_cities_near_california = [
    ('Las Vegas', 36.1699, -115.1398,50),
    ('Henderson', 36.0395, -114.9817,30),
    ('Reno', 39.5296, -119.8138,30),
    ('North Las Vegas', 36.1989, -115.1175,30),
    ('Sparks', 39.5349, -119.7539,30),
    ('Carson City', 39.1638, -119.7674,30),
    ('Boulder City', 35.5414, -114.8340,30),
    ('Pahrump', 36.2084, -116.0246,30),
    ('Elko', 40.8329, -115.7630,180),
    ('Ely', 39.2519, -114.8815,100),
    ('Mesquite', 36.8076, -113.0621,30),
    ('Stateline', 38.9614, -119.9446,30),
    ('Winnemucca', 40.9714, -117.7357,130),
    ('Fallon', 39.4747, -118.7752,40),
    ('Bullhead City',35.1397,-114.5286,30),
    ('Panaca',37.4291,-114.0847,120),
    ('Lake Mead National Recreation Area', 36.2327, -114.5561,80), 
    ('Kingman', 35.1894, -114.0530,80),
    ('Lake Mead',36.2340,-114.6349,100),
    ('Montañas',37.32953,-116.11474,30),    
    ('Desert National',36.74635,-115.36217,100),
    ('Sheldon',41.84533,-119.57550,50),
    ('McDermitt', 41.99247, -117.72431,100),
    ('Sulpur', 40.88059, -118.72330,50),
    ('Lovelock', 40.17077,-118.46512,50),
    ('Austin',39.50010,-117.06437,80),
    ('Fallon',39.46618,-118.77274,65),
    ('Warm Springs',38.18237,-116.36673,50),
    ('Pahrump' ,36.20747,-115.98290,10),
    ('Manhathan',38.53826,-117.07604,80),
    ('Adaven',38.12028,-115.60387,80),
    ('Beatty',36.90684,-116.754669,30),
    ('Ash Meadows',36.40502,-116.32073,30),
    ('Black Rock', 41.20023,-119.02396,40),
    ('Fernley',39.60238,-119.26016,40),
    ('Gerlach', 40.61903,-119.34805,50),
    ('high rock',41.30476,-119.30097,60),
    ('Gerlach', 40.61903,-119.34805,50),
    ('pyramid lake', 40.11891,-119.68039,30),
    ('Carson city',39.17146,-119.73532,30),
    ('Goldfield',37.70433,-117.21396,30),
    ('Yerington',38.97873,-119.17454,50),
    ('Hawthorne',38.52678,-118.62110,50),
    ('Gabbs',38.85050,-117.93171,80),
    ('columbus',38.10657,-118.11573,80),
    ('Tonopah', 38.04626,-117.21619,70),
    ('red rock hounds',39.84207,-119.93413,50)    
    ]

    for city, latitude, longitude,r in nevada_cities_near_california:

        blythe_location=(latitude,longitude)
        # Definir el rango de distancia en kilómetros (por ejemplo, 10 km)
        distance_threshold = r
        # Filtrar los registros que están fuera del rango de distancia
        california_df = california_df[~california_df.apply(is_within_range, location=blythe_location, threshold=distance_threshold, axis=1)]
    generar_mapa(california_df,'google')
    return california_df

def lectura_de_yelp():
    global df_yelp 
    # Especifica la ruta de tu archivo .pkl
    file_path = 'Data-origen/business.pkl'
    #file_path = '/opt/airflow/source-data/business.pkl'
    
    # Lee el archivo .pkl y crea un DataFrame
    df_yelp = pd.read_pickle(file_path)
    return df_yelp

def limpieza_de_yelp():
    global df_yelp
    df_yelp = df_yelp.loc[:,~df_yelp.columns.duplicated()]
    df_yelp=df_yelp.loc[:,['business_id','name', 'latitude', 'longitude','categories','review_count','stars']]
    # Filtra el DataFrame para que solo incluya registros dentro de los límites de California
    california_yelp = df_yelp[(df_yelp['latitude'] >= california_lat_min) & (df_yelp['latitude'] <= california_lat_max) &
                   (df_yelp['longitude'] >= california_lon_min) & (df_yelp['longitude'] <= california_lon_max)]
    nevada_cities_near_california = [    
    ('Reno', 39.5296, -119.8138,30)
    ]
    for city, latitude, longitude,r in nevada_cities_near_california:

        blythe_location=(latitude,longitude)
        # Definir el rango de distancia en kilómetros (por ejemplo, 10 km)
        distance_threshold = r
        # Filtrar los registros que están fuera del rango de distancia
        df_yelp = california_yelp[~california_yelp.apply(is_within_range, location=blythe_location, threshold=distance_threshold, axis=1)]
    generar_mapa(df_yelp,'yelp')
    return 
        

def generar_mapa(dataset,nombre):
    # Crear un mapa centrado en los Estados Unidos
    m = folium.Map(location=[37.0902, -95.7129], zoom_start=5)
    # Preparar los datos para el mapa de calor
    heat_data = [[row['latitude'], row['longitude']] for index, row in dataset.iterrows()]
    # Añadir el mapa de calor
    HeatMap(heat_data).add_to(m)
    # Guardar el mapa en un archivo HTML
    filename='heatmap_'+nombre+'.html'
    if os.path.exists(filename):
        os.remove(filename)
    m.save(filename)
    return
    
def most_common_category(categories):
    return Counter(categories).most_common(1)[0][0]

def union_de_data():
    global df_yelp
    global california_df
    global df_combined
    california_df=california_df.loc[:,['gmap_id','name','latitude','longitude','category','num_of_reviews','avg_rating']]
    
    df_yelp = df_yelp.rename(columns={
    'business_id': 'gmap_id',
    'categories': 'category',
    'review_count': 'num_of_reviews',
    'stars': 'avg_rating'
    })
    print("yelp",df_yelp.shape)
    print("california",california_df.shape)
    df_combined = pd.concat([california_df, df_yelp], ignore_index=True)
    print("combinado ",df_combined.shape)
    
    
    #df_combined.drop_duplicates(inplace=True)
    df_combined=df_combined.dropna()
    df_combined = df_combined.loc[~df_combined.index.duplicated(keep='first')]
    
    df_combined['combined_categories'] = df_combined['category'].apply(lambda x: ' '.join(x))

    # Vectorizar las categorías combinadas
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df_combined['combined_categories'])
    # Aplicar KMeans
    kmeans = KMeans(n_clusters=2)
    df_combined['cluster'] = kmeans.fit_predict(X)

    # # Asignar la categoría más común dentro de cada cluster como la categoría principal
    df_combined['primary_category'] = df_combined.groupby('cluster')['combined_categories'].transform(lambda x: most_common_category(' '.join(x).split()))

    #categorias con modelo k-means
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df_combined['combined_categories'])
    # Selecciona el número de clústeres (por ejemplo, 2)
    num_clusters = 21
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    # Asigna cada descripción a un clúster
    df_combined['cluster'] = kmeans.labels_
    
    print("union de datasets")
    return df_combined
  
def preprocess_text(text,stop_words):
    # Tokenizar el texto
    words = word_tokenize(text.lower())
    # Filtrar palabras vacías
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return words


def clasificacion_final():
    global df_combined
    
    stop_words = set(stopwords.words('english'))
    grouped = df_combined.groupby('cluster')
    # Inicializar un diccionario para almacenar las palabras más comunes por clúster
    common_words_by_cluster = {}

    # Iterar sobre cada grupo (clúster)
    for cluster, group in grouped:
        all_words = []
        for text in group['combined_categories']:  # Suponiendo que 'categories' contiene el texto
            words = preprocess_text(text,stop_words)
            all_words.extend(words)
        # Contar las palabras y obtener las más comunes
        common_words = Counter(all_words).most_common(10)
        common_words_by_cluster[cluster] = common_words

    most_common_word_by_cluster = {}

    for cluster, group in grouped:
        all_words = []
        for text in group['combined_categories']:  # Suponiendo que 'categories' contiene el texto
            words = preprocess_text(text,stop_words)
            all_words.extend(words)
        # Contar las palabras y obtener la más común
        if all_words:
            most_common_word = Counter(all_words).most_common(1)[0][0]
            # Guardar en el diccionario con el formato 'cluster_palabra'
            #most_common_word_by_cluster[f'{cluster}_{most_common_word}'] = most_common_word
            most_common_word_by_cluster[cluster] = most_common_word  
    df_combined['primary_category'] = df_combined['cluster'].map(most_common_word_by_cluster)    
    df_combined.loc[df_combined['primary_category']=='restaurant'].to_parquet('restaurantes_california.parquet', engine='pyarrow')
    print(df_combined.shape)
    print(df_combined.columns)
    return df_combined

lectura_de_google()
limpieza_de_google()
lectura_de_yelp()
limpieza_de_yelp()
union_de_data()
clasificacion_final()