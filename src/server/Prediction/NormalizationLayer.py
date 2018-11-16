import torch

class NormalizationLayer(torch.nn.Module):

    def __init__(self, features, eps=1e-6):

        super(NormalizationLayer, self).__init__()

        self.gamma = torch.nn.Parameter(torch.ones(features))
        self.beta = torch.nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):

        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
