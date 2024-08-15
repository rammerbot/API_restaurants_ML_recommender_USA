from ml import get_recommendations, data, model, pd
from sklearn.metrics.pairwise import cosine_similarity

# Sistema de Recomendacion
def recommender(restaurant: str):
   
    recommendations = get_recommendations(name=restaurant, data=data)
    return recommendations

# Sistema de prediccion. 
def predictor(county,cluster,
            negative_sentiment,positive_sentiment,
            data=data):
    
    # Agrupar por categoria y id de la categoria
    counties = data.groupby(['county_id','COUNTY_NAM']).size().reset_index(name='counts')

    # pasar categoria a diccionario
    counties_dic = counties.set_index('COUNTY_NAM')['county_id'].to_dict()

    # Agrupar por categoria y id de la categoria
    cat_grouped = data.groupby(['cluster', 'cluster_categories']).size().reset_index(name='counts')

    # pasar categoria a diccionario
    cat_dic = cat_grouped.set_index('cluster_categories')['cluster'].to_dict()

    # Pasar ciudades a minusculas
    
    info = [
    counties_dic[county],
    cat_dic[cluster],
    negative_sentiment,
    positive_sentiment,
    ]
    data = pd.DataFrame([info])
    prediction = model.predict(data)
    prediction = int(prediction)
    return prediction

