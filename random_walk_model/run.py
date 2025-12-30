import pickle
import networkx as nx
import numpy as np
from scipy import sparse

PATH_DATA = './data/clean_data.pkl'
k_val = 50

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


def run_rwr(G, start_matrix, alpha=0.85, max_iter=50, tol=1e-6):
    nodes = list(G.nodes())
    n = len(nodes)

    A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight='weight', format='csr')

    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    P = sparse.diags(1.0 / row_sums) @ A

    S = start_matrix.copy()
    Restart = start_matrix.copy()

    for i in range(max_iter):
        S_prev = S.copy()

        S = alpha * P.dot(S) + (1 - alpha) * Restart
        err = np.abs(S - S_prev).mean()
        if err < tol:
            print(f"Converged after {i+1} iterations.")
            break

    return S, nodes


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
    ITEM_OFFSET = n_customers


    ### Instantiate a networkx graph
    G = nx.Graph()
    G.add_nodes_from(range(n_customers))
    G.add_nodes_from(range(ITEM_OFFSET, ITEM_OFFSET + n_products))
    for i in range(0, len(dataset_dictionary['train_edges'])):
        u, v = dataset_dictionary['train_edges'][i]
        w = dataset_dictionary['train_ratings'][i]
        G.add_weighted_edges_from([(u, ITEM_OFFSET + v, w)])


    ### Every Customer is identified as an initial starting point for Random Walk
    mat1 = np.eye(n_customers)
    mat2 = np.zeros((n_products, n_customers))

    initial_matrix = np.concatenate((mat1, mat2), axis=0)

    final_matrix, _ = run_rwr(G, initial_matrix)

    similarity_matrix = final_matrix.T[:, 2100:]

    train_edges = np.array(dataset_dictionary['train_edges']).T
    model_recommendations = get_top_k_recommendations(similarity_matrix, train_edges, k=k_val)

    ### Calculating Recall@K (K=50)
    test_set = [set() for _ in range(n_users)]
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

    

