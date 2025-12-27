import pickle
import matplotlib.pyplot as plt

import torch
import transformers
import torch_geometric.transforms as T

from helper import get_hetero_graph_data, train, evaluate
from model import Model, classifier


PATH_DATA = './data/clean_data.pkl'
PATH_MODEL = './gnn_model/outputs/model_weights.pth'
PATH_LOSS_PLOT = './gnn_model/outputs/loss_plot.png'

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


    transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.0,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=1.0,
        add_negative_train_samples=False,
        edge_types=("user", "rates", "item"),
        rev_edge_types=("item", "rev_rates", "user"),
        is_undirected=False
    )

    train_data, val_data, _ = transform(graph_data)

    train_data, val_data = train_data.to(device), val_data.to(device)

    train_loss_values = []
    val_loss_values = []

    model = Model(hidden_dim=256, user_input_dim=512, item_input_dim=769, metadata=graph_data.metadata())

    max_val_roc = 0

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1, 501):
        train_loss = train(model, optimizer, train_data)
        val_loss, val_auc = evaluate(model, val_data)
        train_loss_values.append(train_loss)
        val_loss_values.append(val_loss)
        if(val_auc > max_val_roc):
            max_val_roc = val_auc
            torch.save(model.state_dict(), PATH_MODEL)
        print('Epoch:', epoch, '||', 'Train Loss:', round(train_loss, 5), '||', 'Val Loss:', round(val_loss, 5), '||', 'Val ROC:', round(val_auc, 5))

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_values, label='Training Loss')
    plt.plot(val_loss_values, label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(PATH_LOSS_PLOT)
    