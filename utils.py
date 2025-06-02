import pandas as pd
import logging
import requests

def load_data(dataset_path, dataset_name):
    """Load dataset based on dataset name and path"""
    if dataset_name == "ml-1m":
        rating = pd.read_csv(dataset_path, sep='::', header=None, 
                           names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
    elif dataset_name == "100k":
        rating = pd.read_csv(dataset_path, sep=",", header=None, 
                           names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
    elif dataset_name == "lastfm-2k":
        rating = pd.read_csv(dataset_path, sep="\t", header=None, 
                           names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
    elif dataset_name == "hetrec":
        rating = pd.read_csv(dataset_path, sep="\t", header=None, 
                           names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
        rating = rating.sort_values(by='uid', ascending=True)
    else:
        # Default format - you can adjust this
        rating = pd.read_csv(dataset_path, sep=",", header=None, 
                           names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
    
    # Reindex users and items (same as in your example)
    user_id = rating[['uid']].drop_duplicates().reindex()
    user_id['userId'] = range(len(user_id))
    rating = pd.merge(rating, user_id, on=['uid'], how='left')
    
    item_id = rating[['mid']].drop_duplicates()
    item_id['itemId'] = range(len(item_id))
    rating = pd.merge(rating, item_id, on=['mid'], how='left')
    
    rating = rating[['userId', 'itemId', 'rating', 'timestamp']]
    
    print(f'Range of userId is [{rating.userId.min()}, {rating.userId.max()}]')
    print(f'Range of itemId is [{rating.itemId.min()}, {rating.itemId.max()}]')

    interactions = list(zip(rating['userId'], rating['itemId'], rating['timestamp']))
    num_user = rating['userId'].nunique()
    num_items = rating['itemId'].nunique()
    
    return interactions, num_user, num_items

def initLogging(logFilename):
    """Init for logging
    """
    logging.basicConfig(
                    level    = logging.DEBUG,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    datefmt  = '%y-%m-%d %H:%M',
                    filename = logFilename,
                    filemode = 'w');
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def send_webhook_message(webhook_url, message, username=None):
    data = {"content": message}
    
    if username:
        data["username"] = username
    
    try:
        response = requests.post(webhook_url, json=data)
        if response.status_code == 204:
            print("Message sent successfully!")
        else:
            print(f"Failed to send message: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")