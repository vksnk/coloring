import torch
import torch.nn.functional as F


def potts_loss(h, edge_index):
    """
    Calculates the Potts loss (inter-node similarity) to penalize similar
    probability distributions between connected nodes.
    """
    # Convert node embeddings/logits to probabilities
    h = F.softmax(h, dim=1)

    # Extract source (u) and target (v) indices from the sparse edge list
    u = edge_index[0]
    v = edge_index[1]

    # Gather the probability distributions for the source and target nodes
    h_u = h[u]
    h_v = h[v]

    # Calculate the inner product (dot product) between neighbors
    # which measures how similar the predictions are for connected nodes
    prod = h_u * h_v
    prod = prod.sum(dim=1)

    # Return the mean similarity (minimizing this encourages connected nodes
    # to have orthogonal/different class assignments)
    return torch.mean(prod)


def entropy_loss(h):
    """
    Computes the mean Shannon entropy of the prediction distributions to
    encourage confident (low-entropy) predictions.
    """
    # Calculate probabilities and log-probabilities
    # (log_softmax is used for numerical stability)
    p = F.softmax(h, dim=1)
    log_p = F.log_softmax(h, dim=1)

    # Calculate entropy: H(p) = - sum(p * log(p))
    entropy = -1 * (p * log_p).sum(dim=1)

    # Return the average entropy across all nodes
    return torch.mean(entropy)
