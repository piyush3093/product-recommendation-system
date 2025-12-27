import pickle

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from helper import get_hetero_graph_data, get_top_k_recommendations
from model import Model

PATH_DATA = './data/clean_data.pkl'
PATH_MODEL = './gnn_model/outputs/model_weights.pth'
k_val = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    graph_data = get_hetero_graph_data(dataset_dictionary, device)

    n_users = dataset_dictionary['n_customers']

    graph_data = graph_data.to(device)

    model = Model(hidden_dim=256, user_input_dim=512, item_input_dim=769, metadata=graph_data.metadata())
    model = model.to(device)
    model.load_state_dict(torch.load(PATH_MODEL, weights_only=True))

    model.eval()
    with torch.no_grad():
        model_out = model(graph_data)
        z_user, z_item = F.normalize(model_out['user']), F.normalize(model_out['item'])
        similarity_matrix = F.sigmoid(torch.mm(z_user, z_item.t())).cpu().numpy()

    train_edges = graph_data['user', 'rates', 'item'].edge_index.detach().cpu().numpy().copy()

    model_recommendations = get_top_k_recommendations(similarity_matrix, train_edges, k=k_val)

    ### Calculating Recall@K (K=50)
    test_set = [set() for _ in range(n_users)]
    for user_idx, product_idx in dataset_dictionary['test_edges']:
        test_set[user_idx].add(product_idx)
    
    recommendation_set = [set(model_recommendations[i].flat) for i in range(n_users)]

    total_count = 0
    total_recall = 0

    for i in range(n_users):
        if(len(test_set[i]) == 0):
            continue
        recommendation_intersection = test_set[i].intersection(recommendation_set[i])
        recall_score = len(recommendation_intersection) / len(test_set[i])
        total_recall += recall_score
        total_count += 1

    print('Recall@K:', total_recall / total_count, 'For K =', k_val)