import torch
import torch.nn.functional as F


def potts_loss(h, edge_index):
    h = F.softmax(h, dim=1)
    u = edge_index[0]
    v = edge_index[1]
    h_u = h[u]
    h_v = h[v]

    prod = h_u * h_v
    prod = prod.sum(dim=1)

    return torch.mean(prod)


def entropy_loss(h):
    p = F.softmax(h, dim=1)
    log_p = F.log_softmax(h, dim=1)

    entropy = -1 * (p * log_p).sum(dim=1)
    return torch.mean(entropy)
