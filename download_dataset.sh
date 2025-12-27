mkdir data
mkdir gnn_model/outputs

###############

wget -P ./data https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Amazon_Fashion.jsonl.gz
wget -P ./data https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Amazon_Fashion.jsonl.gz

###############

gzip -d ./data/Amazon_Fashion.jsonl.gz
gzip -d ./data/meta_Amazon_Fashion.jsonl.gz