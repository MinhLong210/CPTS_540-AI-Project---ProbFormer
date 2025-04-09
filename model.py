import torch
import torch.nn as nn
from torch.distributions import Normal

class ProbFormer(nn.Module):
    def __init__(self, vit, sigma_init=0.1, num_classes=10):
        super().__init__()
        self.vit = vit
        self.d_model = vit.hidden_dim  # 768
        self.n_heads = vit.encoder.layers[0].self_attention.num_heads  # 12
        self.d_k = self.d_model // self.n_heads  # 64
        self.num_layers = len(vit.encoder.layers)  # 12
        self.sigmas = nn.ParameterList([nn.Parameter(torch.tensor(sigma_init)) 
                                       for _ in range(self.num_layers)])
        # self.classifier = vit.heads.head

        # New classification head for CIFAR-10
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.d_model, 256),  # [768, 256]
        #     nn.ReLU(),
        #     nn.Linear(256, num_classes),  # [256, 10]
        # )
        
        self.classifier = nn.Linear(self.d_model, num_classes)  # [768, 10]
        

    def build_pc_layer_batched(self, x, mha, sigma):
        batch_size, n, _ = x.shape  # [B, n, d_model]
        W_qkv = mha.in_proj_weight  # [3*d_model, d_model]
        W_o = mha.out_proj.weight   # [d_model, d_model]
        W_q, W_k, W_v = W_qkv.chunk(3, dim=0)
        bias_qkv = mha.in_proj_bias  # [3*d_model] or None
        bias_o = mha.out_proj.bias   # [d_model] or None

        # Compute Q, K, V for all heads at once (with biases)
        Q = torch.matmul(x, W_q.T)
        K = torch.matmul(x, W_k.T)
        V = torch.matmul(x, W_v.T)
        if bias_qkv is not None:
            b_q, b_k, b_v = bias_qkv.chunk(3, dim=0)
            Q = Q + b_q.view(1, 1, -1)
            K = K + b_k.view(1, 1, -1)
            V = V + b_v.view(1, 1, -1)
        Q = Q.view(batch_size, n, self.n_heads, self.d_k).permute(0, 2, 1, 3)  # [B, H, n, d_k]
        K = K.view(batch_size, n, self.n_heads, self.d_k).permute(0, 2, 1, 3)  # [B, H, n, d_k]
        V = V.view(batch_size, n, self.n_heads, self.d_k).permute(0, 2, 1, 3)  # [B, H, n, d_k]

        # Attention weights (batched over all tokens and heads)
        # attn_scores = torch.einsum('bnhd,bmkd->bnmh', Q, K) / (self.d_k ** 0.5)  # [B, n, n, H]
        # attn_weights = torch.softmax(attn_scores, dim=2)  # [B, n, n, H]
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_k ** 0.5)  # [B, H, n, n]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, H, n, n]


        # Leaf nodes v^i_k: Define distribution (Gaussian) for V
        v_means = V  # [B, H, n, d_k], pretrained means from ViT
        v_vars = torch.ones_like(v_means) * torch.exp(sigma)  # [B, H, n, d_k], variance per v^i_k
        v_dists = Normal(v_means, torch.sqrt(v_vars))  # Batched distribution for v^i_k

        # Sum node: Head outputs as mixture of v^i_k
        # head_means = torch.einsum('bnmh,bmhd->bnhd', attn_weights, v_means)  # [B, n, H, d_k]
        head_means = torch.matmul(attn_weights, v_means)  # [B, H, n, d_k]
        # Variance of the mixture: sum w_k * Var(v^i_k) + w_k * (v^i_k - mean)^2
        # head_vars = torch.einsum('bnmh,bmhd->bnhd', attn_weights, v_vars)  # [B, n, H, d_k]
        head_vars = torch.matmul(attn_weights, v_vars) # [B, H, n, d_k]
        v_deviations = (v_means - head_means) ** 2  # [B, H, n, d_k]
        head_vars += torch.einsum('bhnm,bhmd->bhnd', attn_weights, v_deviations)  # Total variance

        # Product node: Concatenate heads into Z_j
        # z_means = head_means.reshape(batch_size, n, self.d_model)  # [B, n, d_model]
        z_means = head_means.permute(0, 2, 1, 3).reshape(batch_size, n, self.d_model)
        # z_vars = head_vars.reshape(batch_size, n, self.d_model)  # [B, n, d_model]
        z_vars = head_vars.permute(0, 2, 1, 3).reshape(batch_size, n, self.d_model)  # [B, n, d_model]
        # import pdb; pdb.set_trace()

        # Output: y_j = Z_j W_O
        y_means = torch.matmul(z_means, W_o.T)  # [B, n, d_model]
        if bias_o is not None:
            y_means = y_means + bias_o.view(1, 1, -1)
        # Approximate y_j variances (diagonal only)
        # Var(y_j) = sum_i W_o[:, i]^2 * Var(Z_j[:, i]), assuming Z_j components independent
        W_o_sq = W_o**2  # [d_model, d_model]
        y_vars = torch.einsum('bnm,md->bnd', z_vars, W_o_sq)  # [B, n, d_model]

        # Distribution with diagonal variances
        y_dists = Normal(y_means, torch.sqrt(y_vars))  # [B, n, d_model]
        
        return y_dists
    
    def forward(self, input):
        x = self.vit._process_input(input) # Patchify: [B, n, d_model]
        # Prepend CLS token: [B, 196, 768] -> [B, 197, 768]
        batch_size = x.shape[0]
        cls_token = self.vit.class_token.expand(batch_size, -1, -1)  # [B, 1, 768]
        x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        # Add positional embeddings
        x = x + self.vit.encoder.pos_embedding
        # Apply initial dropout
        x = self.vit.encoder.dropout(x)

        for l, layer in enumerate(self.vit.encoder.layers):
            residual = x
            x = layer.ln_1(x)
            # x, att_weights = layer.self_attention(x, x, x)
            # import pdb; pdb.set_trace()
            y_dists = self.build_pc_layer_batched(x, layer.self_attention, self.sigmas[l])
            x = layer.dropout(y_dists.loc) + residual # This is deterministic
            # x = layer.dropout(y_dists.rsample()) + residual # This allows uncertainty from sampling with reparam trick
            residual = x
            x = layer.ln_2(x)
            x = layer.mlp(x) + residual
        x = self.vit.encoder.ln(x) # last layer norm

        # Classification
        cls_output = x[:, 0]  # [B, 768]
        logits = self.classifier(cls_output)  # [B, 10]

        # Transform CLS token distribution to logits space
        cls_mean = y_dists.loc[:, 0]  # [B, 768]
        cls_var = y_dists.scale[:, 0]**2  # [B, 768]
        logits_mean = torch.matmul(cls_mean, self.classifier.weight.T) + self.classifier.bias  # [B, 10]
        logits_var = torch.einsum('bm,nm->bn', cls_var, self.classifier.weight**2)  # [B, 10]
        cls_dist = Normal(logits_mean, torch.sqrt(logits_var))  # [B, 10]

        return logits, cls_dist