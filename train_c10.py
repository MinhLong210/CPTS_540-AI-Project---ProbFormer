import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.distributions import Normal
import numpy as np
from tqdm import tqdm
import random
import csv

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disables optimization that might introduce randomness

set_seed(42)  # Call this before any random operations

# HYPERPARAMS
LEARNING_RATE_SIGMA = 1e-3
LEARNING_RATE_HEAD = 0.01
SIGMA_INIT = 0.1
ALPHA_WEIGHT = 0

class ProbFormer(nn.Module):
    def __init__(self, vit, sigma_init=0.1, num_classes=10):
        super().__init__()
        self.vit = vit
        self.d_model = vit.hidden_dim  # 768
        self.n_heads = vit.encoder.layers[0].self_attention.num_heads  # 12
        self.d_k = self.d_model // self.n_heads  # 64
        self.num_layers = len(vit.encoder.layers)  # 12
        self.sigmas = nn.ParameterList([nn.Parameter(torch.ones(1) * sigma_init) 
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

        # Sum node: Compute head outputs as mixture means
        # head_means = torch.einsum('bnmh,bmhd->bnhd', attn_weights, V)  # [B, n, H, d_k]
        head_means = torch.matmul(attn_weights, V)  # [B, H, n, d_k]
        # Add uncertainty (sigma) to head outputs
        head_vars = torch.ones_like(head_means) * sigma**2  # [B, H, n, d_k]

        # Product node: Concatenate heads into Z_j
        # z_means = head_means.reshape(batch_size, n, self.d_model)  # [B, n, d_model]
        z_means = head_means.permute(0, 2, 1, 3).reshape(batch_size, n, self.d_model)
        # import pdb; pdb.set_trace()

        # Output: y_j = Z_j W_O
        y_means = torch.matmul(z_means, W_o.T)  # [B, n, d_model]
        if bias_o is not None:
            y_means = y_means + bias_o.view(1, 1, -1)

        # Approximate y_j variances (diagonal only)
        # Var(y_j) = sum_i W_o[:, i]^2 * Var(Z_j[:, i]), assuming Z_j components independent
        # z_vars = head_vars.reshape(batch_size, n, self.d_model)  # [B, n, d_model]
        z_vars = head_vars.permute(0, 2, 1, 3).reshape(batch_size, n, self.d_model)  # [B, n, d_model]
        W_o_sq = W_o**2  # [d_model, d_model]
        y_vars = torch.einsum('bnm,md->bnd', z_vars, W_o_sq)  # [B, n, d_model]

        # Distribution with diagonal variances
        y_dists = Normal(y_means, torch.sqrt(y_vars))  # [B, n, d_model]
        
        return y_dists
    
    def build_pc_layer_batched_correct_version(self, x, mha, sigma):
        batch_size, n, _ = x.shape  # [B, n, d_model]
        W_qkv = mha.in_proj_weight  # [3*d_model, d_model]
        W_o = mha.out_proj.weight   # [d_model, d_model]
        W_q, W_k, W_v = W_qkv.chunk(3, dim=0)  # Each [d_model, d_model]
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

        # Attention computation
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_k ** 0.5)  # [B, H, n, n]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, H, n, n]
        head_outputs = torch.matmul(attn_weights, V)  # [B, H, n, d_k]
        head_outputs = head_outputs.permute(0, 2, 1, 3).reshape(batch_size, n, self.d_model)  # [B, n, d_model]
        attn_output = head_outputs

        # Apply output projection
        output = torch.matmul(attn_output, W_o.T)  # [B, n, d_model]
        if bias_o is not None:
            output = output + bias_o.view(1, 1, -1)

        return output

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
            x = layer.dropout(y_dists.loc) + residual
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
    
    # def forward(self, x): #### 90+ % accuracy
    #     x = self.vit._process_input(x)  # Patchify: [B, 196, d_model]
    #     # Prepend CLS token: [B, 196, 768] -> [B, 197, 768]
    #     batch_size = x.shape[0]
    #     cls_token = self.vit.class_token.expand(batch_size, -1, -1)  # [B, 1, 768]
    #     x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
    #     x = self.vit.encoder(x)  # Transformer layers
    #     cls_output = x[:, 0]  # CLS token: [B, 768]
    #     logits = self.classifier(cls_output)  # [B, 10]
    #     return logits, None

# CIFAR-10 DataLoaders
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit = models.vit_b_16(pretrained=True)
model = ProbFormer(vit, sigma_init=SIGMA_INIT, num_classes=10).to(device)
optimizer = torch.optim.Adam([
    {'params': model.sigmas.parameters(), 'lr': LEARNING_RATE_SIGMA},
    {'params': model.classifier.parameters(), 'lr': LEARNING_RATE_HEAD}
])
criterion = nn.CrossEntropyLoss()

# Evaluation function
def evaluate(model, dataloader, dataset_name):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Evaluating {dataset_name}", leave=False):
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Finetuning loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, cls_dist = model(images)
        cls_loss = criterion(logits, labels)
        nll_loss = -cls_dist.log_prob(logits).mean()  # Simplified NLL
        loss = cls_loss + ALPHA_WEIGHT * nll_loss

        # loss = cls_loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    train_acc = evaluate(model, trainloader, "Train")
    test_acc = evaluate(model, testloader, "Test")
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(trainloader):.4f}, "
          f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    
    with open(f"logs/training_log_lrsigma{LEARNING_RATE_SIGMA}_lrhead{LEARNING_RATE_HEAD}_alpha{ALPHA_WEIGHT}.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        
        # Write the header if the file is empty
        if file.tell() == 0:
            writer.writerow(["Epoch", "Loss", "Train Accuracy (%)", "Test Accuracy (%)"])
        
        # Write the training data
        writer.writerow([epoch + 1, round(running_loss / len(trainloader), 4), round(train_acc, 2), round(test_acc, 2)])

torch.save(model.state_dict(), f"logs/probformer_cifar10_lrsigma{LEARNING_RATE_SIGMA}_lrhead{LEARNING_RATE_HEAD}_alpha{ALPHA_WEIGHT}.pth")