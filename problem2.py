import numpy as np
import pandas as pd
import gensim.downloader as gensim_api
from scipy.stats import pearsonr

model = gensim_api.load('glove-wiki-gigaword-300')

categories = {
    'animals/size': {
        'positive': ['large', 'big', 'huge'],
        'negative': ['small', 'little', 'tiny'],
        'words': ['alligator', 'ant', 'bee', 'bird', 'butterfly', 'camel', 'cheetah',
                  'chicken', 'chipmunk', 'crow', 'dog', 'dolphin', 'duck', 'elephant',
                  'goldfish', 'hamster', 'hawk', 'horse', 'mammoth', 'monkey', 'moose',
                  'mosquito', 'mouse', 'orca', 'penguin', 'pig', 'rhino', 'salmon',
                  'seal', 'snake', 'swordfish', 'tiger', 'turtle', 'whale']
    },
    'clothing/wealth': {
        'positive': ['rich', 'wealthy', 'privileged'],
        'negative': ['poor', 'poverty', 'underprivileged'],
        'words': ['bathrobe', 'belt', 'bikini', 'blouse', 'boots', 'bra', 'bracelet',
                  'coat', 'collar', 'cuff', 'dress', 'earrings', 'glasses', 'gloves',
                  'gown', 'hat', 'jacket', 'jeans', 'knickers', 'loafers', 'necklace',
                  'nightgown', 'overcoat', 'pajamas', 'panties', 'pants', 'pantyhose',
                  'raincoat', 'robe', 'sandals', 'scarf', 'shawl', 'shirt', 'shorts',
                  'skirt', 'sleeve', 'slippers', 'sneakers', 'socks', 'stockings',
                  'sweater', 'sweatshirt', 'swimsuit', 'thong', 'tiara', 'tights',
                  'trousers', 'tuxedo', 'vest', 'watch']
    }
}

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

def project_words(words, dimension):
    projected_values = []
    for word in words:
        if word in model:
            projected_value = np.dot(model[word], dimension) 
            projected_values.append(projected_value)
    return projected_values

def load_ratings(filename):
    ratings_df = pd.read_csv(filename, header=None, names=['item', 'rating'])  
    return ratings_df

def get_ratings(words, ratings_df):
    ratings = []
    for word in words:
        rating = ratings_df[ratings_df['item'] == word]['rating']
        if not rating.empty:
            ratings.append(rating.values[0])  
        else:
            ratings.append(0)  
    return ratings

def calculate_correlation(projected_values, ratings):
    correlation, _ = pearsonr(projected_values, ratings)
    return correlation


def analyze_category(category, positive_adjectives, negative_adjectives, words, ratings_file):
    dimension = compute_dimension(positive_adjectives, negative_adjectives)
    projected_values = project_words(words, dimension)
    ratings_df = load_ratings(ratings_file)
    ratings = get_ratings(words, ratings_df)
    correlation = calculate_correlation(projected_values, ratings)
    return correlation


print("Problem 2A Results:")
for category, data in categories.items():
    correlation = analyze_category(
        category,
        data['positive'],
        data['negative'],
        data['words'],
        f"problem2_ratings_{'as' if 'animals' in category else 'cw'}.csv"
    )
    print(f"Correlation for {category}: {correlation:.3f}")
    
    
categories_2b = {
    'animals/size': {
        'positive': ['enormous', 'gigantic', 'colossal'],
        'negative': ['miniature', 'microscopic', 'diminutive'],
        'words': categories['animals/size']['words']
    },
    'clothing/wealth': {
        'positive': ['affluent', 'opulent', 'prosperous'],
        'negative': ['destitute', 'impoverished', 'indigent'],
        'words': categories['clothing/wealth']['words']
    }
}

print("Problem 2B Results:")
for category, data in categories_2b.items():
    correlation = analyze_category(
        category,
        data['positive'],
        data['negative'],
        data['words'],
        f"problem2_ratings_{'as' if 'animals' in category else 'cw'}.csv"
    )
    print(f"Correlation for {category}: {correlation:.3f}")