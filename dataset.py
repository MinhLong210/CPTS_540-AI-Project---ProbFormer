import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# CIFAR-10
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# CIFAR-100
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset_c100 = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform)
trainloader_c100 = torch.utils.data.DataLoader(trainset_c100, batch_size=128, shuffle=True, num_workers=2)
testset_c100 = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform)
testloader_c100 = torch.utils.data.DataLoader(testset_c100, batch_size=128, shuffle=False, num_workers=2)