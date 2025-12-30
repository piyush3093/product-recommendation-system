import torch
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
from torch_geometric.utils import negative_sampling
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import roc_auc_score

from model import classifier

def get_sentence_embeddings(sentence_list, device=torch.device('cuda')):

    sentence_model = SentenceTransformer('all-distilroberta-v1', device = device)

    with torch.no_grad():
        product_embeddings = sentence_model.encode(sentence_list, normalize_embeddings=True)

    return torch.from_numpy(product_embeddings)



def get_hetero_graph_data(dataset, device):

    """
    This function takes in a dataset as a dictionary with keys:
    1. products: title or description (string) of each product
    2. product_rating: rating (integer) of each product
    3. n_customers: total customers
    4. train_edges: User-Product Links
    5. train_ratings: User-Product Rating
    
    It returns a heterogenous graph dataset with sentence embeddings for each product as its features
    For each user it returns a uniform feature vector
    Links are constructed using interactions between user and product.
    """

    product_embeddings = get_sentence_embeddings(dataset['products'], device)
    ratings_embeddings = (torch.tensor(dataset['product_rating']) / 5.0).view(-1, 1)
    product_embeddings = torch.cat((product_embeddings, ratings_embeddings), dim = 1)
    user_embeddings = torch.ones(dataset['n_customers'], 512, dtype=torch.float32) / 512

    data = HeteroData()

    data['user'].x = user_embeddings  # [num_users, dim_u]
    data['item'].x = product_embeddings  # [num_items, dim_i]

    all_edge_indexes = torch.tensor(np.array(dataset['train_edges']).T, dtype=torch.long)
    all_edge_ratings = torch.tensor(np.array(dataset['train_ratings']), dtype=torch.float32)

    data['user', 'rates', 'item'].edge_index = all_edge_indexes
    data['item', 'rev_rates', 'user'].edge_index = all_edge_indexes.flip(0)
    data['user', 'rates', 'item'].edge_weight = all_edge_ratings
    data['item', 'rev_rates', 'user'].edge_weight = all_edge_ratings

    return data


def train(model, optimizer, data):

    model.train()

    neg_edge_index = negative_sampling(
        edge_index=data["user", "rates", "item"].edge_index,
        num_nodes=(data['user'].x.shape[0], data['item'].x.shape[0]),
        num_neg_samples=data['user', 'rates', 'item'].edge_label.shape[0],
        method='sparse' # 'sparse' is generally better for large, sparse graphs
    )

    optimizer.zero_grad()
    model_out = model(data)
    z_user = F.normalize(model_out['user'])
    z_item = F.normalize(model_out['item'])
    pos_edges, neg_edges = data['user', 'rates', 'item'].edge_label_index, neg_edge_index
    edge_label_index = torch.cat([pos_edges, neg_edges], dim=1)
    edge_labels = torch.cat([torch.ones(pos_edges.shape[1]), torch.zeros(neg_edges.shape[1])]).to(z_user.device)
    loss = classifier(z_user, z_item, edge_label_index, edge_labels)
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate(model, data):

    model.eval()

    with torch.no_grad():
        model_out = model(data)
        z_user = F.normalize(model_out['user'])
        z_item = F.normalize(model_out['item'])
        edge_label_index = data['user', 'rates', 'item'].edge_label_index
        edge_labels = data['user', 'rates', 'item'].edge_label
        loss = classifier(z_user, z_item, edge_label_index, edge_labels)
        pred = (z_user[edge_label_index[0]] * z_item[edge_label_index[1]]).sum(dim=-1)

    pred = pred.cpu().numpy()
    ground_truths = edge_labels.cpu().numpy()
    auc = roc_auc_score(ground_truths, pred)

    return loss.item(), auc


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