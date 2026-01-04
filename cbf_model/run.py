import pickle
import numpy as np

import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import csr_matrix, issparse


PATH_DATA = './data/clean_data.pkl'
k_val = 50

nltk.download('stopwords')
nltk.download('wordnet')

def get_top_k_recommendations(prob_matrix, train_edges, k=10):

    """
    prob_matrix: np.ndarray of shape (n_users, n_products)
    train_edges: list of tuples or 2D array [(u1, p1), (u2, p2), ...]
    k: number of products to recommend per user
    
    This function returns top k product recommendations for all users 
    excluding the products already seen in train edges

    masking ensures train user-product probability pairs are set to -1
    so they do not appear among recommendations

    np.argpartition is used to partition topk probabilities
    """
    
    masked_probs = prob_matrix.copy()
    user_indices, product_indices = train_edges
    masked_probs[user_indices, product_indices] = -1.0
    n_products = prob_matrix.shape[1]
    top_k_indices_unsorted = np.argpartition(masked_probs, -k, axis=1)[:, -k:]

    return top_k_indices_unsorted


def custom_preprocessor(text):
    """
    Performs text cleaning (lowercase, punctuation/digit removal), tokenization, stop word removal, and lemmatization.
    """

    lemmatizer = WordNetLemmatizer()
    english_stopwords = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text) # remove punctuation
    text = re.sub(r'\d+', '', text) # remove digits
    tokens = text.split()

    processed_tokens = []
    for word in tokens:
        if word not in english_stopwords:
            word = lemmatizer.lemmatize(word)
            processed_tokens.append(word)

    return ' '.join(processed_tokens)



if __name__ == '__main__':

    try:
        with open(PATH_DATA, 'rb') as f:
            dataset_dictionary = pickle.load(f)
        print("Dictionary successfully loaded from pickle file:")
        print(dataset_dictionary.keys())
    except FileNotFoundError:
        print(f"Error: The file '{PATH_DATA}' was not found.")
    except Exception as e:
        print(f"An error occurred while loading the dictionary: {e}")

    
    n_customers = dataset_dictionary['n_customers']
    n_products = len(dataset_dictionary['products'])

    cleaned_description = list(map(custom_preprocessor, dataset_dictionary['products']))

    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.002, max_df=0.9, stop_words=None)
    X_products = tfidf.fit_transform(cleaned_description) ### (n_products, n_features)
    if not issparse(X_products):
        X_products = csr_matrix(X_products)

    ### Sparse matrix of shape (n_users, n_products) 
    ### that contains rating of each respective product reviewed by user
    user_item_matrix = csr_matrix(
        (np.array(dataset_dictionary['train_ratings']), np.array(dataset_dictionary['train_edges']).T),
        shape=(n_customers, n_products)
    )

    X_users = user_item_matrix.dot(X_products)
    user_total_weights = user_item_matrix.sum(axis=1)
    user_total_weights[user_total_weights == 0] = 1e-9
    user_total_weights_dense = user_total_weights.A
    X_users = X_users.multiply(1 / user_total_weights_dense) ### Taking weighted average of each product rated
    X_users = csr_matrix(X_users)

    if issparse(X_products):
        X_products_T = X_products.transpose().tocsr()
    else:
        X_products_T = X_products.T

    R_scores = X_users.dot(X_products_T)

    similarity_matrix = R_scores.toarray()

    train_edges = np.array(dataset_dictionary['train_edges']).T
    model_recommendations = get_top_k_recommendations(similarity_matrix, train_edges, k=k_val)

    ### Calculating Recall@K (K=50)
    test_set = [set() for _ in range(n_customers)]
    for user_idx, product_idx in dataset_dictionary['test_edges']:
        test_set[user_idx].add(product_idx)

    recommendation_set = [set(model_recommendations[i].flat) for i in range(n_customers)]

    total_count = 0
    total_recall = 0

    for i in range(n_customers):
        if(len(test_set[i]) == 0):
            continue
        recommendation_intersection = test_set[i].intersection(recommendation_set[i])
        recall_score = len(recommendation_intersection) / len(test_set[i])
        total_recall += recall_score
        total_count += 1

    print('Recall@K:', total_recall / total_count, 'For K =', k_val)