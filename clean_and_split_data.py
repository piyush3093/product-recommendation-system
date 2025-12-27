import pandas as pd
import numpy as np
import json
import random, math
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split


PATH_REVIEWS = "./data/Amazon_Fashion.jsonl"
PATH_PRODUCTS = "./data/meta_Amazon_Fashion.jsonl"
PATH_OUTPUT = "./data/clean_data.pkl"
MIN_PRODUCT_COUNT = 10


if __name__ == '__main__':

    reviews = []

    with open(PATH_REVIEWS, "r") as file:
        lines = file.readlines()

    for line in lines:
        if line.strip():  # Avoid parsing empty lines
            reviews.append(json.loads(line))



    meta = []

    with open(PATH_PRODUCTS, "r") as file:
        lines = file.readlines()

    for line in lines:
        if line.strip():  # Avoid parsing empty lines
            meta.append(json.loads(line))


    print('Total Reviews:', len(reviews), '\nTotal Products:', len(meta), '\n')


    customer_count = dict()
    product_count = dict()

    for i in range(0, len(reviews)):
        current_user = reviews[i]['user_id']
        if(current_user in customer_count):
            customer_count[current_user] += 1
        else:
            customer_count[current_user] = 1

    for i in range(0, len(meta)):
        current_product = meta[i]['parent_asin']
        if(current_product in product_count):
            product_count[current_product] += 1
        else:
            product_count[current_product] = 1

    print('Total Customers:', len(customer_count), '\nTotal Products:', len(product_count), '\n')


    #### Only keep customers with >= 10 reviews

    final_customers = set()

    for user in customer_count:
        if(customer_count[user] >= MIN_PRODUCT_COUNT):
            final_customers.add(user)

    final_products = set()

    for i in range(0, len(reviews)):
        current_user = reviews[i]['user_id']
        current_product = reviews[i]['parent_asin']
        if(current_user not in final_customers):
            continue
        if(current_product in final_products):
            continue
        final_products.add(current_product)

    print('Filtered Customers:', len(final_customers), '\nFiltered Products:', len(final_products), '\n')


    #### Create hashmaps to provide unique integer ids to Customers and Products

    customer_hashmap = {}
    product_hashmap = {}

    count = 0
    for customer_id in final_customers:
        customer_hashmap[customer_id] = count
        count += 1

    count = 0
    for product_id in final_products:
        product_hashmap[product_id] = count
        count += 1
    
    #### Save item titles and ratings in separate lists

    product_text = [None for _ in range(len(product_hashmap))]
    product_ratings = [None for _ in range(len(product_hashmap))]

    for i in range(0, len(meta)):
        current_product = meta[i]['parent_asin']
        description = meta[i]['title']
        rating = meta[i]['average_rating']
        if(current_product not in final_products):
            continue
        product_hash = product_hashmap[current_product]
        product_text[product_hash] = description
        product_ratings[product_hash] = rating

    all_edges = []
    all_ratings = []

    for i in range(0, len(reviews)):
        current_user = reviews[i]['user_id']
        current_product = reviews[i]['parent_asin']
        current_rating = reviews[i]['rating']
        if(current_user not in final_customers):
            continue
        customer_hash = customer_hashmap[current_user]
        product_hash = product_hashmap[current_product]
        all_edges.append([customer_hash, product_hash])
        all_ratings.append(current_rating)

    train_edges, test_edges, train_ratings, test_ratings = train_test_split(all_edges, all_ratings, test_size=0.1, random_state=102)

    ### the final cleaned dataset is stored in a dictionary
    ### products: each item's title, product_rating: each item's rating
    ### train_edges: links in train set, train_ratings: rating of review
    ### test_edges: links in test set
    ### n_customers: total customers

    data = {'products': product_text, 'product_rating': product_ratings, 
            'train_edges': train_edges, 'train_ratings': train_ratings,
            'test_edges': test_edges, 'test_ratings': test_ratings, 
            'n_customers': len(final_customers)}

    
    ### Save the data to a pickle file

    try:
        with open(PATH_OUTPUT, 'wb') as file:
            pickle.dump(data, file)
        print(f"Dictionary successfully saved to {PATH_OUTPUT}")
    except Exception as e:
        print(f"Error saving dictionary: {e}")

    