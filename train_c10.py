import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from tqdm import tqdm
import random
import csv
from model import ProbFormer
from dataset import *
from utils import *
from config import *

set_seed(42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit = models.vit_b_16(pretrained=True)
model = ProbFormer(vit, sigma_init=SIGMA_INIT, num_classes=10).to(device)
optimizer = torch.optim.Adam([
    {'params': model.sigmas.parameters(), 'lr': LEARNING_RATE_SIGMA},
    {'params': model.classifier.parameters(), 'lr': LEARNING_RATE_HEAD}
])
criterion = nn.CrossEntropyLoss()

# Finetuning loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, cls_dist = model(images)
        cls_loss = criterion(logits, labels)
        nll_loss = -cls_dist.log_prob(logits).mean()
        # print(nll_loss)
        loss = cls_loss + ALPHA_WEIGHT * nll_loss

        loss.backward()
        # print(np.mean([model.sigmas[l].item() for l, _ in enumerate(model.vit.encoder.layers)]))
        optimizer.step()
        running_loss += loss.item()
    
    train_acc, train_nll, train_ece = evaluate(model, trainloader, "Train")
    test_acc, test_nll, test_ece = evaluate(model, testloader, "Test")
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(trainloader):.4f}, "
          f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    
    with open(f"logs/training_log_lrsigma{LEARNING_RATE_SIGMA}_lrhead{LEARNING_RATE_HEAD}_alpha{ALPHA_WEIGHT}.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        
        if file.tell() == 0:
            writer.writerow(["Epoch", "Loss", "train_acc", "train_nll", "train_ece", "test_acc", "test_nll", "test_ece"])
        
        writer.writerow([epoch + 1, round(running_loss / len(trainloader), 4), 
                         round(train_acc, 2), round(train_nll, 2), round(train_ece, 2), 
                         round(test_acc, 2), round(test_nll, 2), round(test_ece, 2)])

torch.save(model.state_dict(), f"logs/probformer_cifar10_lrsigma{LEARNING_RATE_SIGMA}_lrhead{LEARNING_RATE_HEAD}_alpha{ALPHA_WEIGHT}.pth")