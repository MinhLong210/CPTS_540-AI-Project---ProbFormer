import torch
import torch.nn as nn
import torchvision.models as models
from torch.distributions import Normal
import numpy as np

# Simulated Probabilistic Circuit classes
class PCNode:
    def __init__(self, scope):
        self.scope = scope

class InputNode(PCNode):
    def __init__(self, scope, distribution):
        super().__init__(scope)
        self.distribution = distribution

    def sample(self, n_samples=1):
        return self.distribution.sample((n_samples,))

    def log_prob(self, x):
        return self.distribution.log_prob(x)

class SumNode(PCNode):
    def __init__(self, scope, children, weights):
        super().__init__(scope)
        self.children = children
        self.weights = weights  # Normalized weights (e.g., attention scores)

    def sample(self, n_samples=1):
        choices = np.random.choice(len(self.children), size=n_samples, p=self.weights)
        samples = torch.stack([self.children[i].sample(1)[0] for i in choices])
        return samples

    def log_prob(self, x):
        log_probs = torch.stack([child.log_prob(x) for child in self.children])
        return torch.logsumexp(torch.log(torch.tensor(self.weights, dtype=torch.float32)) + log_probs, dim=0)

    def mean(self):
        # Approximate mean of the mixture
        return sum(w * child.distribution.loc for w, child in zip(self.weights, self.children))

    def covariance(self):
        # Approximate covariance of the mixture (assuming independence for simplicity)
        mean = self.mean()
        var = sum(w * (child.distribution.scale**2 + (child.distribution.loc - mean)**2) 
                  for w, child in zip(self.weights, self.children))
        return torch.diag(var)

class ProductNode(PCNode):
    def __init__(self, scope, children):
        super().__init__(scope)
        self.children = children

    def sample(self, n_samples=1):
        samples = [child.sample(n_samples) for child in self.children]
        return torch.cat(samples, dim=-1)

    def log_prob(self, x):
        split_sizes = [child.mean().shape[-1] for child in self.children]
        x_split = torch.split(x, split_sizes, dim=-1)
        return sum(child.log_prob(x_part) for child, x_part in zip(self.children, x_split))

    def mean(self):
        return torch.cat([child.mean() for child in self.children], dim=-1)

    def covariance(self):
        # Block-diagonal covariance (independent heads)
        covs = [child.covariance() for child in self.children]
        return torch.block_diag(*covs)

# Load pretrained ViT
vit = models.vit_b_16(pretrained=True)
vit.eval()

# Extract first Transformer layer's MHA
mha = vit.encoder.layers[0].self_attention
n_heads = mha.num_heads  # H
d_model = vit.hidden_dim  # e.g., 768
d_k = d_model // n_heads  # e.g., 64

# Example input: Batch of 1 image (3 channels, 224x224)
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    x_patches = vit._process_input(x)  # [1, n+1, d_model]
    n = x_patches.shape[1]  # e.g., 197
    x_mha_input = vit.encoder.layers[0].ln_1(x_patches)

# Access MHA weights
W_qkv = mha.in_proj_weight  # [3*d_model, d_model]
W_o = mha.out_proj.weight   # [d_model, d_model]
W_q, W_k, W_v = W_qkv.chunk(3, dim=0)

# Build PC for one token (j=1) across all heads
j = 1  # Token index
sigma = 0.1  # Variance for uncertainty
product_children = []

for i in range(n_heads):
    # Compute Q, K, V for head i
    start_idx = i * d_k
    end_idx = (i + 1) * d_k
    Q_i = torch.matmul(x_mha_input, W_q[start_idx:end_idx, :].T)  # [1, n, d_k]
    K_i = torch.matmul(x_mha_input, W_k[start_idx:end_idx, :].T)  # [1, n, d_k]
    V_i = torch.matmul(x_mha_input, W_v[start_idx:end_idx, :].T)  # [1, n, d_k]
    
    # Attention weights
    attn_scores_i = torch.matmul(Q_i, K_i.transpose(-1, -2)) / (d_k ** 0.5)
    attn_weights_i = torch.softmax(attn_scores_i, dim=-1)
    A_ij = attn_weights_i[0, j, :]  # [n]

    # Leaf nodes: v^i_k
    leaf_nodes_i = [InputNode(scope=[V_i[0, k, :]], distribution=Normal(V_i[0, k, :], sigma)) 
                    for k in range(n)]
    
    # Sum node children: p(V^i_j | v^i_k, X)
    sum_children_i = [InputNode(scope=[f"V^{i+1}_{j}"], 
                               distribution=Normal(leaf_nodes_i[k].distribution.loc, sigma)) 
                      for k in range(n)] # auxilliary variable V^i_j to make sure the sum nodes children have the same scope
    
    # Sum node: head^i_j
    head_node_i = SumNode(scope=[f"head^{i+1}_{j}"], children=sum_children_i, weights=A_ij)
    product_children.append(head_node_i)
    import pdb; pdb.set_trace()

# Product node: Z_j
product_node = ProductNode(scope=[f"Z_{j}"], children=product_children)

# Final output: y_j = Z_j W_O
z_mean = product_node.mean()  # [d_model]
z_cov = product_node.covariance()  # [d_model, d_model]
y_mean = torch.matmul(z_mean, W_o.T)  # [d_model]
y_cov = W_o.T @ z_cov @ W_o  # [d_model, d_model]
y_dist = Normal(y_mean, torch.sqrt(torch.diagonal(y_cov)))  # Diagonal approximation

# Example usage
n_samples = 5
y_samples = y_dist.sample((n_samples,))
print("Samples of y_j:", y_samples)
print("Log prob of first sample:", y_dist.log_prob(y_samples[0]))
