import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

transforms = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])


def get_accuracy(model, data_loader):
    correct_pred = 0
    n = 0
    with torch.no_grad():
        model.eval()
        for x, y_true in data_loader:
            x, y_true = x.to(device), y_true.to(device)

            pred = model(x)
            _, predicted_labels = torch.max(pred, 1)
            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()
    return correct_pred.float() / n


def train(train_loader, model, criterion, optimizer):
    model.train()
    running_loss = 0
    for x, y_true in train_loader:
        x, y_true = x.to(device), y_true.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y_true)
        running_loss += loss.item() * x.size(0)
        loss.backward()
        optimizer.step()
    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss


def test(test_loader, model, criterion):
    model.eval()
    running_loss = 0
    for x, y_true in test_loader:
        x, y_true = x.to(device), y_true.to(device)

        pred = model(x)
        loss = criterion(pred, y_true)
        running_loss += loss.item() * x.size(0)
    epoch_loss = running_loss / len(test_loader.dataset)
    return model, epoch_loss


def training_loop(model, criterion, optimizer, train_loader, test_loader, epochs, print_every=1):
    train_losses = []
    test_losses = []
    for epoch in range(0, epochs):
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer)
        train_losses.append(train_loss)
        with torch.no_grad():
            model, test_loss = test(test_loader, model, criterion)
            torch.save(model.state_dict(), os.path.join("./lab7data/model/Net-{}.pt".format(epoch + 1)))
            test_losses.append(test_loss)
        if epoch % print_every == (print_every - 1):
            train_acc = get_accuracy(model, train_loader)
            test_acc = get_accuracy(model, test_loader)
            print(f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Test loss: {test_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Test accuracy: {100 * test_acc:.2f}')
    return model, optimizer, (train_losses, test_losses)


class LeNet(nn.Module):
    def __init__(self, n_classes):
        super(LeNet, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        pred = self.classifier(x)
        return pred


def main():
    print("PyTorch version: " + torch.__version__)
    random_seed = 1
    learning_rate = 0.001
    batch_size = 32
    n_classes = 10
    n_epochs = 10

    # download and create datasets
    train_dataset = datasets.MNIST(root='./lab7data/mnist_data', train=True, transform=transforms, download=True)
    test_dataset = datasets.MNIST(root='./lab7data/mnist_data', train=False, transform=transforms)

    # define the data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    torch.manual_seed(random_seed)
    model = LeNet(n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, test_loader, n_epochs)


if __name__ == "__main__":
    main()
