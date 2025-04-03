import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class ProbabilisticCircuit(nn.Module):
    def __init__(self, input_dim=784, num_products=16, num_sums=32, num_classes=10):
        super(ProbabilisticCircuit, self).__init__()
        
        self.input_dim = input_dim
        self.num_products = num_products
        self.num_sums = num_sums
        self.num_classes = num_classes
        
        # Leaf nodes: Gaussian parameters for each input dimension
        self.means = nn.Parameter(torch.randn(input_dim))
        self.log_vars = nn.Parameter(torch.zeros(input_dim))
        
        # Product node grouping
        self.group_size = input_dim // num_products
        
        # Sum node weights (learnable parameters for weighted sums)
        self.sum_weights = nn.Parameter(torch.ones(num_products, num_sums) / num_sums)
        
        # Output layer weights (from sum nodes to classes)
        self.output_weights = nn.Parameter(torch.ones(num_sums, num_classes) / num_classes)
        
    def _leaf_nodes(self, x):
        # Compute log-probabilities for Gaussian leaf nodes
        x = x.view(x.size(0), -1)  # Flatten input [batch_size, input_dim]
        log_probs = -0.5 * ((x - self.means) ** 2 / torch.exp(self.log_vars) + self.log_vars + torch.log(torch.tensor(2 * torch.pi)))
        return log_probs  # [batch_size, input_dim]
    
    def _product_nodes(self, leaf_probs):
        # Product nodes: multiply probabilities (sum in log-space)
        product_probs = []
        for i in range(0, self.input_dim, self.group_size):
            group_probs = leaf_probs[:, i:i + self.group_size].sum(dim=1)  # Sum log probs
            product_probs.append(group_probs)
        return torch.stack(product_probs, dim=1)  # [batch_size, num_products]
    
    def _sum_nodes(self, product_probs):
        # Sum nodes: weighted sum over product nodes
        # Normalize weights to ensure they sum to 1
        normalized_weights = torch.softmax(self.sum_weights, dim=1)  # [num_products, num_sums]
        sum_probs = torch.matmul(product_probs, normalized_weights)  # [batch_size, num_sums]
        return sum_probs
    
    def _output_layer(self, sum_probs):
        # Output layer: weighted sum to class probabilities
        normalized_weights = torch.softmax(self.output_weights, dim=1)  # [num_sums, num_classes]
        output_probs = torch.matmul(sum_probs, normalized_weights)  # [batch_size, num_classes]
        return output_probs
    
    def forward(self, x):
        # Forward pass through the probabilistic circuit
        leaf_probs = self._leaf_nodes(x)
        product_probs = self._product_nodes(leaf_probs)
        sum_probs = self._sum_nodes(product_probs)
        output_probs = self._output_layer(sum_probs)
        return output_probs  # Return class probabilities

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
learning_rate = 0.001
batch_size = 64
num_epochs = 10

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(
    root='../data',
    train=True,
    transform=transform,
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    root='../data',
    train=False,
    transform=transform
)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss, and optimizer
model = ProbabilisticCircuit().to(device)
criterion = nn.CrossEntropyLoss()  # Suitable for classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

# Testing function
def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100 * correct / total
    return test_loss, test_acc

# Training loop
print("Starting training...")
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = test(model, test_loader, criterion, device)
    
    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
    print('------------------------')

# Save the trained model
torch.save(model.state_dict(), 'probabilistic_circuit_mnist.pth')
print("Training completed and model saved!")