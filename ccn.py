import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalContrastiveNetwork(nn.Module):
    def __init__(self, embedding_net, n_conditions, embedding_size, projection_size):
        """ embeddingnet: The network that projects the inputs into an embedding of embedding_size
            n_conditions: Integer defining number of different similarity notions
            embedding_size: Number of dimensions of the embedding output from the embeddingnet
            learnedmask: Boolean indicating whether masks are learned or fixed
            prein: Boolean indicating whether masks are initialized in equally sized disjoint 
                sections or random otherwise"""
        super(ConditionalContrastiveNetwork, self).__init__()

        self.n_conditions = n_conditions
        self.embedding_net = embedding_net
        self.embedding_size = embedding_size
        self.projection_size = projection_size

        # backbone and projection heads
        self.heads = []
        for i in range(self.n_conditions):
            self.heads.append(torch.nn.Sequential(
                torch.nn.Linear(self.embedding_size, self.embedding_size),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(self.embedding_size, self.projection_size)
            ))

    def forward(self, x):
        embed = self.embedding_net(x)
        feats = []
        for i in range(self.n_conditions):
            feats.append(F.normalize(self.heads[i](embed), dim=1))
        return feats
