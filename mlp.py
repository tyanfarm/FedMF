import torch
from engine import Engine
from utils import use_cuda, resume_checkpoint


class MLP(torch.nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.embedding_user = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.logistic = torch.nn.Sigmoid()

    def forward(self, item_indices):
        user_embedding = self.embedding_user(torch.LongTensor([0 for i in range(len(item_indices))]).cuda())
        item_embedding = self.embedding_item(item_indices)
        logits = (user_embedding * item_embedding).sum(dim=1, keepdim=True)  # element-wise product
        logits = logits.view(-1)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass

class MLPEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        self.model = MLP(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(MLPEngine, self).__init__(config)
        print(self.model)
