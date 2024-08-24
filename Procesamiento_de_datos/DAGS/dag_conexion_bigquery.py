from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from google.cloud import bigquery
import pandas as pd

# Configuración de los parámetros
GCP_PROJECT_ID = 'robust-radar-431706-u1'
BIGQUERY_DATASET = 'negocios'
BIGQUERY_TABLE = 'restaurantes'

# Crear un DataFrame de ejemplo ['gmap_id','name','latitude','longitude','category','num_of_reviews','avg_rating']
def create_dataframe():
    data = {'gmap_id': [1, 2, 3],
            'name': ['a', 'b', 'c'],
            'latitude':[33.3456,33.3456,33.3456],
            'longitude':[-91.2345,-91.2345,-91.2345],
            'category':['restaurantes','restaurantes2','restaurantes3'],
            'num_of_reviews':[2,3,4],
            'avg_rating':[3,4,5]
            }
    df = pd.DataFrame(data)
    return df

# Función para cargar el DataFrame a BigQuery
def load_dataframe_to_bigquery(**kwargs):
    df = kwargs['ti'].xcom_pull(task_ids='create_dataframe_task')

    # Configuración de la tabla en BigQuery
    client = bigquery.Client(project=GCP_PROJECT_ID)
    table_id = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_TABLE}"
    
    # Configuraciones adicionales para la carga
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,  # Reemplaza la tabla si ya existe
    )
    
    # Cargar el DataFrame a BigQuery
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()  # Espera a que el job se complete

# Definir el DAG
default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
    'retries': 1,
}

dag = DAG(
    'load_dataframe_to_bigquery',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
)

# Definir la tarea para crear el DataFrame
create_dataframe_task = PythonOperator(
    task_id='create_dataframe_task',
    python_callable=create_dataframe,
    dag=dag,
)

# Definir la tarea para cargar el DataFrame a BigQuery
load_to_bigquery_task = PythonOperator(
    task_id='load_to_bigquery_task',
    python_callable=load_dataframe_to_bigquery,
    provide_context=True,
    dag=dag,
)

# Configurar el orden de las tareas
create_dataframe_task >> load_to_bigquery_task
