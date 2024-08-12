from ml import get_recommendations, data
from sklearn.metrics.pairwise import cosine_similarity

# Sistema de Recomendacion
def recommender(movie: str):
    title = movie
    recommendations = get_recommendations(title, data)
    return recommendations
   