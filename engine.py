import numpy as np
from collections import defaultdict
import random
from sklearn.metrics import ndcg_score
from utils import *

class FedMF:
    def __init__(self, num_users, num_items, num_factors=50, learning_rate=0.01, reg=0.01):
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.num_clients = num_users  # Each user is a client
        
        # Initialize global item factors (server-side)
        self.item_factors = np.random.normal(0, 0.1, (num_items, num_factors))
        
        # Initialize user factors for each client (each client is one user)
        self.client_user_factors = {}
        self.client_users = {}
        
        # Assign each user to their own client
        for user_id in range(num_users):
            self.client_users[user_id] = [user_id]  # Each client has exactly one user
            self.client_user_factors[user_id] = np.random.normal(0, 0.1, (1, num_factors))  # 1 user per client

    def split_data(self, interactions):
        """
        Split data into train, validation, and test sets.
        Removes the latest item from user_interactions after assigning to test set.
        interactions: list of (user_id, item_id, timestamp) tuples
        Returns user_interactions (with latest item removed), val_data, test_data
        """
        # Sort interactions by timestamp
        interactions = sorted(interactions, key=lambda x: x[2])
        
        # Group interactions by user
        user_interactions = defaultdict(list)
        for u, i, t in interactions:
            user_interactions[u].append((i, t))
            
        val_data = []
        test_data = []
        all_items = set(range(self.num_items))
        
        for user in user_interactions:
            # Sort items by timestamp for this user
            items = sorted(user_interactions[user], key=lambda x: x[1], reverse=True)
            
            # Test set: latest item + 99 negative samples
            if len(items) >= 1:
                test_data.append((user, items[0][0], 1))  # Positive sample
                non_interacted = list(all_items - set(i for i, _ in items))
                negative_samples = random.sample(non_interacted, min(99, len(non_interacted)))
                test_data.extend([(user, item, 0) for item in negative_samples])
                # Remove the latest item from user_interactions
                user_interactions[user] = [item for item in user_interactions[user] if item != items[0]]
            
            # Validation set: second latest item + 99 negative samples
            if len(items) >= 2:
                val_data.append((user, items[1][0], 1))  # Positive sample
                negative_samples = random.sample(non_interacted, min(99, len(non_interacted)))
                val_data.extend([(user, item, 0) for item in negative_samples])
                # Remove the second latest item from user_interactions
                user_interactions[user] = [item for item in user_interactions[user] if item != items[1]]
                
        return user_interactions, val_data, test_data

    def sample_train_data(self, user_interactions, client_id):
        """
        Sample training data for a specific client (single user), taking remaining items after val/test split
        and 4 negative samples per positive item.
        """
        train_data = []
        all_items = set(range(self.num_items))
        user = self.client_users[client_id][0]  # Client has one user
        
        if user not in user_interactions:
            return train_data
            
        # Get items after excluding the latest 2 (used for test and val)
        items = sorted(user_interactions[user], key=lambda x: x[1], reverse=True)[2:]
        non_interacted = list(all_items - set(i for i, _ in user_interactions[user]))
        
        for item, _ in items:
            train_data.append((user, item, 1))  # Positive sample
            if len(non_interacted) >= 4:
                negative_samples = random.sample(non_interacted, 4)
                train_data.extend([(user, item, 0) for item in negative_samples])
                
        return train_data

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, user_interactions, val_data, test_data, num_epochs=10, batch_size=256, msg_discord=None):
        """
        Train the model in a federated manner using mini-batch SGD.
        Each client (user) updates their own user factors locally and contributes item factor gradients to the server.
        """
        best_val = 0
        best_test = 0
        best_ndcg = 0
        
        for epoch in range(num_epochs):
            # Server initializes item gradients
            global_item_grads = np.zeros_like(self.item_factors)
            # Sample 10% of clients to simulate partial participation
            # active_clients = random.sample(list(range(self.num_clients)), max(1, int(0.1 * self.num_clients)))
            active_clients = list(range(self.num_clients))
            random.shuffle(active_clients)
            
            for client_id in active_clients:
                # Sample training data for this client
                train_data = self.sample_train_data(user_interactions, client_id)
                if not train_data:
                    continue
                random.shuffle(train_data)
                
                # Convert to numpy arrays for batching
                train_data = np.array(train_data, dtype=object)
                num_batches = int(np.ceil(len(train_data) / batch_size))
                
                user = self.client_users[client_id][0]  # Single user for this client
                
                for batch_idx in range(num_batches):
                    batch = train_data[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                    users = batch[:, 0].astype(int)
                    items = batch[:, 1].astype(int)
                    labels = batch[:, 2].astype(float)
                    
                    user_vecs = self.client_user_factors[client_id][0]  # Single user vector
                    item_vecs = self.item_factors[items]
                    
                    # Compute predictions
                    preds = np.sum(user_vecs * item_vecs, axis=1)
                    errors = (labels - self.sigmoid(preds)).reshape(-1, 1)
                    
                    # Compute gradients
                    user_grads = errors * item_vecs - self.reg * user_vecs
                    item_grads = errors * user_vecs - self.reg * item_vecs
                    
                    # Update local user factors
                    self.client_user_factors[client_id][0] += self.learning_rate * 80 * np.mean(user_grads, axis=0)
                    
                    # Accumulate item gradients for server aggregation
                    for idx, item in enumerate(items):
                        global_item_grads[item] += self.learning_rate * 80 * item_grads[idx]
            
            # Server updates item factors (average gradients across active clients)
            self.item_factors +=  (global_item_grads / max(1, len(active_clients)))
            
            # Evaluate on validation and test sets
            val_metrics = self.evaluate(val_data)
            test_metrics = self.evaluate(test_data)
            
            logging.info(f"Epoch {epoch+1}: Val HR@10: {val_metrics['hr@10']:.4f}, "
                  f"Val NDCG@10: {val_metrics['ndcg@10']:.4f}, "
                  f"Test HR@10: {test_metrics['hr@10']:.4f}, "
                  f"Test NDCG@10: {test_metrics['ndcg@10']:.4f}")
            
            if val_metrics['hr@10'] > best_val:
                best_val = val_metrics['hr@10']
                best_test = test_metrics['hr@10']
                best_ndcg = test_metrics['ndcg@10']

        WEBHOOK_URL = "https://discord.com/api/webhooks/1379111626933932152/bA5gUgoF8L0RnnrdtFnC6-D8gM2lNbinqeHtuDUEItu0nWeTww9s_ho6fjZhupK8eGR7"
        msg_discord += f"```Best HR@10: {best_test:.4f}, NDCG@10: {best_ndcg:.4f}```"
        send_webhook_message(WEBHOOK_URL, msg_discord, username="Notification Bot")
        
        logging.info(f"Best HR@10: {best_test:.4f}, NDCG@10: {best_ndcg:.4f}")

    def predict(self, user, items, client_id):
        """Predict scores for a user and list of items"""
        if user != self.client_users[client_id][0]:
            return np.zeros(len(items))  # Return zero scores if user not in client
        scores = np.dot(self.client_user_factors[client_id][0], self.item_factors[items].T)
        return scores

    def evaluate(self, data):
        """
        Evaluate model with HR@10 and NDCG@10
        """
        user_items = defaultdict(list)
        for user, item, rating in data:
            user_items[user].append((item, rating))
        
        hr_scores = []
        ndcg_scores = []
        
        for user in user_items:
            # Each user is their own client
            client_id = user
            if client_id not in self.client_users:
                continue
                
            items = [item for item, _ in user_items[user]]
            true_ratings = [rating for _, rating in user_items[user]]
            
            if not any(true_ratings):  # Skip if no positive items
                continue
                
            pred_scores = self.predict(user, items, client_id)
            
            # Get top-10 predicted items
            top_10_indices = np.argsort(pred_scores)[::-1][:10]
            top_10_items = [items[i] for i in top_10_indices]
            top_10_true = [true_ratings[i] for i in top_10_indices]
            
            # HR@10
            hit = any(r == 1 for r in top_10_true)
            hr_scores.append(1 if hit else 0)
            
            # NDCG@10
            ndcg = ndcg_score([true_ratings], [pred_scores], k=10)
            ndcg_scores.append(ndcg)
        
        return {
            'hr@10': np.mean(hr_scores) if hr_scores else 0.0,
            'ndcg@10': np.mean(ndcg_scores) if ndcg_scores else 0.0
        }