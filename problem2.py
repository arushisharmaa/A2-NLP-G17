import numpy as np
import pandas as pd
import gensim.downloader as gensim_api
from scipy.stats import pearsonr

model = gensim_api.load('glove-wiki-gigaword-300')

def load_data(filename):
    with open(filename, 'r') as file:
        data = file.read().strip().split('\n\n') 
    categories = {}
    
    for section in data:
        lines = section.strip().split('\n')
        category_name = lines[0].strip()  
        positive_adjectives = eval(lines[1].strip().split(':')[1].strip()) 
        negative_adjectives = eval(lines[2].strip().split(':')[1].strip())  
        category_words = eval(lines[3].strip().split(':')[1].strip()) 
        categories[category_name] = {
            'positive': positive_adjectives,
            'negative': negative_adjectives,
            'words': category_words
        }
    return categories

categories = load_data('problem2_data.txt')

def compute_dimension(positive_adjectives, negative_adjectives):
    positive_vectors_list = []
    negative_vectors_list = []

    for adj in positive_adjectives:
        if adj in model:
            positive_vectors_list.append(model[adj])

    for adj in negative_adjectives:
        if adj in model:
            negative_vectors_list.append(model[adj])

    positive_vectors = np.mean(positive_vectors_list, axis=0)
    negative_vectors = np.mean(negative_vectors_list, axis=0)
    
    return positive_vectors - negative_vectors

size_dimension = compute_dimension(categories['animals/size']['positive'], categories['animals/size']['negative'])
wealth_dimension = compute_dimension(categories['clothing/wealth']['positive'], categories['clothing/wealth']['negative'])

def project_words(words, dimension):
    projected_values = []
    for word in words:
        if word in model:
            projected_value = np.dot(model[word], dimension) 
            projected_values.append(projected_value)
    return projected_values

projected_animals = project_words(categories['animals/size']['words'], size_dimension)
projected_clothing = project_words(categories['clothing/wealth']['words'], wealth_dimension)

def load_ratings(filename):
    ratings_df = pd.read_csv(filename, header=None, names=['item', 'rating'])  
    return ratings_df

ratings_size = load_ratings('problem2_ratings_as.csv') 
ratings_wealth = load_ratings('problem2_ratings_cw.csv')  

def get_ratings(words, ratings_df):
    ratings = []
    for word in words:
        rating = ratings_df[ratings_df['item'] == word]['rating']
        if not rating.empty:
            ratings.append(rating.values[0])  
        else:
            ratings.append(0)  
    return ratings

size_ratings = get_ratings(categories['animals/size']['words'], ratings_size)
wealth_ratings = get_ratings(categories['clothing/wealth']['words'], ratings_wealth)

def calculate_correlation(projected_values, ratings):
    correlation, _ = pearsonr(projected_values, ratings)
    return correlation

correlation_animals_size = calculate_correlation(projected_animals, size_ratings)
correlation_clothing_wealth = calculate_correlation(projected_clothing, wealth_ratings)

print(f"Correlation for animals and size: {correlation_animals_size:.3f}")
print(f"Correlation for clothing and wealth: {correlation_clothing_wealth:.3f}")
