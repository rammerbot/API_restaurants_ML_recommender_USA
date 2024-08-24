from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os
import json
import pandas as pd
from geopy.distance import great_circle
#import folium
#from folium.plugins import HeatMap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.cluster import KMeans
import nltk
from collections import Counter
nltk.download('stopwords')
nltk.download('stopwords',download_dir='/tmp/nltk_data')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from google.cloud import storage

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import gcsfs
import pandas as pd
from google.cloud import bigquery

GCS_BUCKET = 'data-ini'
GCS_FILE_PATH = 'metadata_300.parquet'
GCS_FILE_PATH_YELP='business.pkl'

# Configuración de los parámetros
GCP_PROJECT_ID = 'robust-radar-431706-u1'
BIGQUERY_DATASET = 'negocios'
BIGQUERY_TABLE = 'restaurantes'

california_lat_min = 32.5343
california_lat_max = 42.0095
california_lon_min = -124.4096
california_lon_max = -114.1315

california_df=pd.DataFrame()
df_empty = pd.DataFrame()
df_combined=pd.DataFrame()
df_yelp=pd.DataFrame()

def is_within_range(row, location, threshold):
    point = (row['latitude'], row['longitude'])
    distance = great_circle(location, point).kilometers
    return distance <= threshold

def lectura_de_google():
    global california_df
    print("inicio de lectura_de_google")
     # Crear un sistema de archivos GCS
    fs = gcsfs.GCSFileSystem()
    print("sistema fs de GCS")
    # Leer el archivo Parquet desde GCS
    with fs.open(f'{GCS_BUCKET}/{GCS_FILE_PATH}', 'rb') as f:
        print("iniciando lectura")
        df = pd.read_parquet(f)
        # Aquí puedes realizar operaciones con el DataFrame
        print(df.head())
        
    print("entro a leer datos de google")
    #gcs_file_path = 'gs://data-ini/google_metadata.parquet'
    #df = pd.read_parquet(gcs_file_path, engine='pyarrow', storage_options={"token": "default"})  
    #print("lectura de archivo")
    df_meta_data=df.loc[:,['gmap_id','address','name', 'latitude', 'longitude','category','num_of_reviews','avg_rating']]
    # Filtra el DataFrame para que solo incluya registros dentro de los límites de California
    california_df = df_meta_data[(df_meta_data['latitude'] >= california_lat_min) & (df_meta_data['latitude'] <= california_lat_max) &
                   (df_meta_data['longitude'] >= california_lon_min) & (df_meta_data['longitude'] <= california_lon_max)]
    print(california_df.shape)
    return california_df

def limpieza_de_google(california_df):
    #global california_df
    # Encontrar filas duplicadas basadas en la columna 'gmap_id'
    #duplicated_rows = california_df[california_df.duplicated(subset='gmap_id', keep=False)]
    # Eliminar duplicados basados en la columna 'gmap_id'
    california_df = california_df.reset_index(drop=True)
    california_df = california_df.drop_duplicates(subset='gmap_id')
    print("drop_duplicates")
    #california_df = california_df.reset_index(drop=True)
    #california_df = california_df.dropna(subset=['address'])
    
    #print("drop_na de address")
    #california_df = california_df.reset_index(drop=True)
    #california_df = california_df.dropna(subset=['category'])
    print("drop_na de category")
    
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
    print("california seccionada por latitud y longitud reg.:",california_df.shape)
    #generar_mapa(california_df,'google')
    return california_df

def lectura_de_yelp():
    global df_yelp 
    #global california_df
    #california_df=california_df
    # Especifica la ruta de tu archivo .pkl
    #file_path = '/opt/airflow/source-data/business.pkl'
    
    #gcs_file_path = 'gs://data-ini/business.pkl'
    #df = pd.read_parquet(gcs_file_path, engine='pyarrow', storage_options={"token": "default"})
    
    fs = gcsfs.GCSFileSystem()
    print("sistema fs de GCS para yelp")
    # Leer el archivo Parquet desde GCS
    with fs.open(f'{GCS_BUCKET}/{GCS_FILE_PATH_YELP}', 'rb') as f:
        print("iniciando lectura yelp")
        df_yelp = pd.read_pickle(f)
        # Aquí puedes realizar operaciones con el DataFrame
        print(df_yelp.head())
        
    # Lee el archivo .pkl y crea un DataFrame
    #file_path = 'dags/source-data/business.pkl'
    #df_yelp = pd.read_pickle(file_path)
    return df_yelp

def limpieza_de_yelp(df_yelp):
    #global df_yelp
    #global california_df
    #california_df=california_df
    try:
        df_yelp = df_yelp.loc[:,~df_yelp.columns.duplicated()]
        print("quitar columnas duplicadas",df_yelp.columns)
        df_yelp=df_yelp.loc[:,['business_id','name', 'latitude', 'longitude','categories','review_count','stars']]
        # Filtra el DataFrame para que solo incluya registros dentro de los límites de California
        print("seleccion de campos")
        california_yelp = df_yelp[(df_yelp['latitude'] >= california_lat_min) & (df_yelp['latitude'] <= california_lat_max) &
                    (df_yelp['longitude'] >= california_lon_min) & (df_yelp['longitude'] <= california_lon_max)]
        nevada_cities_near_california = [    
        ('Reno', 39.5296, -119.8138,30)
        ]
        print("segmentacion por latitud y longitud")
        for city, latitude, longitude,r in nevada_cities_near_california:
            blythe_location=(latitude,longitude)
            # Definir el rango de distancia en kilómetros (por ejemplo, 10 km)
            distance_threshold = r
            # Filtrar los registros que están fuera del rango de distancia
            df_yelp = california_yelp[~california_yelp.apply(is_within_range, location=blythe_location, threshold=distance_threshold, axis=1)]
        print ("yelp seccionada")
    except Exception as e:
        print("Ocurrió un error:",e)
    #generar_mapa(df_yelp,'yelp')
    return df_yelp
        

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

def union_de_data(california_df,df_yelp):
    #global df_yelp
    #global california_df
    global df_combined
    #global california_df
    #california_df=california_df
    #try:
    print("seleccion de columnas con transformacion de california_df",california_df.shape)
    california_df=california_df.loc[:,['gmap_id','name','latitude','longitude','category','num_of_reviews','avg_rating']]
    california_df['combined_categories'] = california_df['category'].apply(lambda x: ' '.join(x))
    california_df['category']=california_df['combined_categories']
    #california_df = california_df.drop(['combined_categories'], axis=1)
    print("renombrar columnas")
    df_yelp = df_yelp.rename(columns={
    'business_id': 'gmap_id',
    'categories': 'category',
    'review_count': 'num_of_reviews',
    'stars': 'avg_rating'
    })
    print("yelp registros",df_yelp.shape)
    print("california",california_df.shape)
    #df_combined = pd.concat([california_df, df_yelp], ignore_index=True)
    df_combined = california_df
    df_combined = df_combined[df_combined['category'].notna() & (df_combined['category'] != '')]
    print("combinado ",df_combined.shape)
    
    
    #df_combined.drop_duplicates(inplace=True)
    df_combined=df_combined.dropna()
    df_combined = df_combined.loc[~df_combined.index.duplicated(keep='first')]
    
    #df_combined['combined_categories'] = df_combined['category'].apply(lambda x: ' '.join(x))

    # Vectorizar las categorías combinadas
    vectorizer = TfidfVectorizer()
    #X = vectorizer.fit_transform(df_combined['combined_categories'])
    X = vectorizer.fit_transform(df_combined['category'])
    
    # Aplicar KMeans
    kmeans = KMeans(n_clusters=2)
    df_combined['cluster'] = kmeans.fit_predict(X)

    # # Asignar la categoría más común dentro de cada cluster como la categoría principal
    df_combined['primary_category'] = df_combined.groupby('cluster')['category'].transform(lambda x: most_common_category(' '.join(x).split()))

    #categorias con modelo k-means
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df_combined['category'])
    # Selecciona el número de clústeres (por ejemplo, 2)
    num_clusters = 25
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    # Asigna cada descripción a un clúster
    df_combined['cluster'] = kmeans.labels_
    
    print("union de datasets")
    #except Exception as e:
    #    print("Ocurrió un error:",e)
    return df_combined
  
def preprocess_text(text,stop_words):
    # Tokenizar el texto
    words = word_tokenize(text.lower())
    # Filtrar palabras vacías
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return words


def clasificacion_final(df_combined):
    #global df_combined
    try:
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
        print("palabras mas comunes")
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
        print("termina for")
        df_combined['primary_category'] = df_combined['cluster'].map(most_common_word_by_cluster)    
        
        #df_combined.loc[df_combined['primary_category']=='restaurant'].to_parquet('restaurantes_california.parquet', engine='pyarrow')
        print(df_combined.shape)
        print(df_combined.columns)
        
        #df = kwargs['ti'].xcom_pull(task_ids='create_dataframe_task')
        print("comienza base de datos")
        # Configuración de la tabla en BigQuery
        client = bigquery.Client(project=GCP_PROJECT_ID)
        table_id = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_TABLE}"
        print("carga de configuracion")
        # Configuraciones adicionales para la carga
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,  # Reemplaza la tabla si ya existe
        )
        print("cargar datos a bd")
        # Cargar el DataFrame a BigQuery
        job = client.load_table_from_dataframe(df_combined, table_id, job_config=job_config)
        print("esperar finalizacion de job")
        job.result()  # Espera a que el job se complete
        
    except Exception as e:
        print("Ocurrió un error:",e)
    return df_combined


def extract(**kwargs):
    # Código de extracción de datos
    df_cal = lectura_de_google()
    kwargs['ti'].xcom_push(key='df_cal', value=df_cal.to_json())
    return df_cal
    

def transform(**kwargs):
    # Código de transformación de datos
    df_json = kwargs['ti'].xcom_pull(key='df_cal', task_ids='extract')
    df_cali = pd.read_json(df_json)  # Reconstruir el DataFrame
    df_cal_result= limpieza_de_google(df_cali)
    kwargs['ti'].xcom_push(key='df_cal_result', value=df_cal_result.to_json())
    return df_cal_result

def load(**kwargs):
    # Código de carga de datos
    # Recuperar el DataFrame desde XCom
    df_cal_result_json = kwargs['ti'].xcom_pull(key='df_cal_result', task_ids='transform')
    df_cal_result = pd.read_json(df_cal_result_json)
    
    # df_cal_result_json = kwargs['ti'].xcom_pull(key='df_cal', task_ids='extract')
    # result = pd.read_json(df_cal_result_json)
    
    df_yel=lectura_de_yelp()
    result=limpieza_de_yelp(df_yel)
    union=union_de_data(df_cal_result,result)
    clasificacion_final(union)
    return

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 8, 7),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
    'catchup': False,
}

dag = DAG(
    dag_id='etl_pipeline_completo_2',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
)

t1 = PythonOperator(
    task_id='extract',
    python_callable=extract,
    dag=dag,
)

t2 = PythonOperator(
    task_id='transform',
    python_callable=transform,
    dag=dag,
)

t3 = PythonOperator(
    task_id='load',
    python_callable=load,
    dag=dag,
)

# scrape_data = PythonOperator(task_id='scrape_{0}'.format(pipeline),
#                                      op_kwargs={'pipeline': pipeline,
#                                                'data_path': data_path,
#                                                'test': test,
#                                                 'headers': headers,
#                                                'crawl_time': crawl_time},
#                                       provide_context=True,
#                                     python_callable=scrape)

t1 >> t2 >> t3

