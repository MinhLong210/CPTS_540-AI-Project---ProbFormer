import torch
import numpy as np
import random
import tqdm

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

def evaluate(model, dataloader, dataset_name, device):
    model.eval()
    correct = 0
    total = 0
    nll_total = 0.0
    confidences = []
    accuracies = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Evaluating {dataset_name}", leave=False):
            images, labels = images.to(device), labels.to(device)
            logits, cls_dist = model(images)  # [B, 10], Normal([B, 10])

            logits = cls_dist.rsample() # Sampling the logits using reparam

            # Accuracy
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # NLL: Compute log-probability of true labels under cls_dist
            # labels_onehot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1)
            nll = -cls_dist.log_prob(logits).mean()  # Simplified NLL using logits as proxy
            nll_total += nll.item() * labels.size(0)

            # ECE: Confidence (softmax probs) vs. accuracy
            probs = torch.softmax(logits, dim=1)  # [B, 10]
            conf, pred = probs.max(1)  # Confidence scores [B]
            accur = (pred == labels).float()  # Binary accuracy [B]
            confidences.append(conf.cpu().numpy())
            accuracies.append(accur.cpu().numpy())

    # Compute metrics
    accuracy = 100 * correct / total
    nll_avg = nll_total / total

    # ECE computation
    confidences = np.concatenate(confidences)
    accuracies = np.concatenate(accuracies)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        if in_bin.sum() > 0:
            avg_conf = confidences[in_bin].mean()
            avg_acc = accuracies[in_bin].mean()
            bin_size = in_bin.sum() / total
            ece += bin_size * abs(avg_conf - avg_acc)

    return accuracy, nll_avg, ece