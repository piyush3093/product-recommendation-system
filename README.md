# Product Recommendation Model

This repository contains Product Recommendation Models based on user preferences. The dataset used is publicly available at https://amazon-reviews-2023.github.io/. <br/>

For computation and simplicity purposes, I have used only fashion subset of the dataset, and filtered out customers who have less than 10 interactions. <br/>

Currently this repository only contains a GNN Model that treats this problem as a link prediction task, however in the future, I will also add a couple more methods based upon Content Based Filtering and Random Walk and compare their performance. <br/>

The data is split in 9:1 ratio for train and test split. For GNNs we further divide train set into validation and supervision subsets and use negative sampling to sample negative edges for training. The model used is a 3-hop Heterogenous GNN. The model and the dataset are small enough so they run easily on Cloud GPUs such as Kaggle or Colab. <br/>

The Metric used for evaluation is Recall@K where K is set to 50.
<ul>
    <li>GNN Model gives a Recall@50 as 0.029.</li>
</ul>

This project was done as part of my Course ELL880 (Special Topics in Computers) under the guidance of our instructor Prof Sougata Mukherjee (IIT Delhi). <br/>

If you want to recreate the results, please first install the required libraries by running these commands on your CLI. <br/>
```
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

pip install sentence-transformers
```

Further you can directly run these commands inorder to download data, clean data, train model and evaluate results.

```
bash download_dataset.sh
python clean_and_split_data.py
python train.py
python eval.py
```

If you want to work on some other Amazon Dataset, please update download_dataset.sh, clean_and_split_data.py accordingly.