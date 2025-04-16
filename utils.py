import torch
import numpy as np
import random
from tqdm import tqdm
from model import DeterministicViT
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

            if isinstance(model, DeterministicViT):
                logits = model(images) 
            else:
                logits, cls_dist = model(images)

            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # nll = -cls_dist.log_prob(logits).mean()
            # nll_total += nll.item() * labels.size(0)
            # if isinstance(model, DeterministicViT):
            #     # Compute probs from logits directly
            #     probs = torch.softmax(logits, dim=1)
            # else:
            #     # If Probformer then use MC sampling to compute probs
            #     # samples = cls_dist.rsample([100])
            #     # probs = torch.softmax(samples, dim=-1).mean(dim=0)
            
            probs = torch.softmax(logits, dim=1)

            
            # NLL
            nll = -torch.log(probs[range(labels.size(0)), labels] + 1e-10)
            nll_total += nll.sum().item()

            conf, pred = probs.max(1)
            accur = (pred == labels).float()
            confidences.append(conf.cpu().numpy())
            accuracies.append(accur.cpu().numpy())

    accuracy = 100 * correct / total
    nll_avg = nll_total / total


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