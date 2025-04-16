import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from tqdm import tqdm
import random
import csv
from model import DeterministicViT
from dataset import *
from utils import evaluate, set_seed
from config import *

set_seed(42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit = models.vit_b_16(pretrained=True)
model = DeterministicViT(vit, num_classes=100).to(device)
optimizer = torch.optim.Adam([
    {'params': model.vit.heads.parameters(), 'lr': LEARNING_RATE_HEAD}
])
criterion = nn.CrossEntropyLoss()

# Finetuning loop
num_epochs = NUM_EPOCHS
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(trainloader_c100, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        cls_loss = criterion(logits, labels)
        # print(nll_loss)
        # Add sigma regularization to encourage non-zero sigmas
        loss = cls_loss
        loss.backward()

        optimizer.step()
        running_loss += loss.item()
    
    train_acc, train_nll, train_ece = evaluate(model, trainloader_c100, "Train", device)
    test_acc, test_nll, test_ece = evaluate(model, testloader_c100, "Test", device)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(trainloader_c100):.4f}, "
          f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    
    with open(f"logs/Cifar100_training_log_DetViT_lrhead{LEARNING_RATE_HEAD}.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        
        if file.tell() == 0:
            writer.writerow(["Epoch", "Loss", "train_acc", "train_nll", "train_ece", "test_acc", "test_nll", "test_ece"])
        
        writer.writerow([epoch + 1, round(running_loss / len(trainloader_c100), 4), 
                         round(train_acc, 2), round(train_nll, 2), round(train_ece, 2), 
                         round(test_acc, 2), round(test_nll, 2), round(test_ece, 2)])

torch.save(model.state_dict(), f"logs/DetViT_cifar100_lrhead{LEARNING_RATE_HEAD}.pth")