from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from querys import ( recommender, predictor
                    )

# instancia de FastApi.
app = FastAPI()

# Version y nombre de API.    
app.title = "Restaurant Recommendation and Prediction - Machine Learning"
app.version = "1.0 Alfa"

# Configurar middleware CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas las solicitudes desde cualquier origen
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (GET, POST, PUT, etc.)
    allow_headers=["*"],  # Permitir todos los encabezados
)

# Recomendar peliculas
@app.get("/recommender/", tags=['Recommend'])
def recomendacion(name: str):
    name= name.lower()
    recomendacion = recommender(name)
    return recomendacion

@app.get("/predictor/", tags=['Predictor'])
def get_predictions(county:str,categoria:str, negativas:int, positivas:int):
 
    prediction = predictor(county, categoria, negativas, positivas)
    
    if prediction == 1:
        return "Predicción: El Negocio tiene alto nivel de éxito."
    else:
        return "Predicción: El negocio tiene un idice bajo de exito."
    
    