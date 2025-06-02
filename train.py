import argparse
import torch 
import torch.nn as nn
import os  
from utils import *
from engine import FedMF
import datetime


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--alias', type=str, default='FedMF')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--l2_reg', type=float, default=1e-3)
parser.add_argument('--num_epochs', type=int, default=300)
parser.add_argument('--dataset', type=str, default='100k')
args = parser.parse_args()

config = vars(args)
if config['dataset'] == 'ml-1m':
    config['num_users'] = 6040
    config['num_items'] = 3706
elif config['dataset'] == '100k':
    config['num_users'] = 943
    config['num_items'] = 1682
elif config['dataset'] == 'lastfm-2k':
    config['num_users'] = 1600
    config['num_items'] = 12454
elif config['dataset'] == 'hetrec':
    config['num_users'] = 2113
    config['num_items'] = 10109
else:
    pass

folders = ["log"]
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Logging.
path = 'log/'
current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logname = os.path.join(path, current_time+'.txt')
initLogging(logname)

# Load dataset
dataset_dir = "data/" + config['dataset'] + "/" + "ratings.dat"
interactions, num_users, num_items = load_data(dataset_dir, config['dataset'])


# Create model
fedmf = FedMF(num_users=num_users, num_items=num_items, num_factors=32, learning_rate=config['lr'], reg=config['l2_reg'])

# Split data
user_interactions, val_data, test_data = fedmf.split_data(interactions)

# Save configuration and results
message_discord = f"\n**Dataset: {config['dataset']}**\n```method: {config['alias']}, lr: {str(config['lr'])}, l2_reg: {str(config['l2_reg'])}, num_epochs: {str(config['num_epochs'])}\n```\n"

# Train model
fedmf.train(user_interactions, val_data, test_data, num_epochs=config['num_epochs'], msg_discord=message_discord)
