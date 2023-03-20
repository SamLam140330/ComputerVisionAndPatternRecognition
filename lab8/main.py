import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as func
from torchvision import datasets, transforms

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

transforms = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
activation = {}


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


def get_activation(name):
    def hook(model, inputs, output):
        activation[name] = output.detach()

    return hook


def main():
    print("PyTorch version: " + torch.__version__)
    n_classes = 10
    row_img = 10
    n_rows = 5

    test_dataset = datasets.MNIST(root='./lab8data/mnist_data', train=False, transform=transforms)
    net = LeNet(n_classes)
    net.load_state_dict(torch.load('./lab8data/model/Net-10.pt'))

    fig = plt.figure()
    for index in range(1, row_img * n_rows + 1):
        plt.subplot(n_rows, row_img, index)
        plt.axis('off')
        plt.imshow(test_dataset.data[index], cmap='gray_r')

        with torch.no_grad():
            net.eval()
            probs = net(test_dataset[index][0].unsqueeze(0))
            probs = func.softmax(probs, 1)
        title = f'{torch.argmax(probs)} ({torch.max(probs * 100):.0f}%)'
        plt.title(title, fontsize=7)

    fig.suptitle('Trained LeNet-predictions')
    conv1 = net.feature_extractor[0]
    conv1.register_forward_hook(get_activation('conv1'))

    # change the index to visualize different images
    data = test_dataset[0][0].unsqueeze(0)
    net(data)
    act = activation['conv1'].squeeze()
    fig, ax_arr = plt.subplots(act.size(0))
    im = None
    for idx in range(act.size(0)):
        im = ax_arr[idx].imshow(act[idx])

    plt.colorbar(im, ax=ax_arr.ravel().tolist(), orientation="horizontal")
    print("Showing the images")
    plt.show()


if __name__ == "__main__":
    main()
