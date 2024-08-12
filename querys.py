from ml import get_recommendations, data
from sklearn.metrics.pairwise import cosine_similarity

# Sistema de Recomendacion
def recommender(restaurant: str):
    name = restaurant
    recommendations = get_recommendations(name, data)
    return recommendations
   