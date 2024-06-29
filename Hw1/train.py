import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm

batchSize = 16
transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batchSize, shuffle=True, num_workers=2
)
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batchSize, shuffle=False, num_workers=2
)


class VGG19BN(nn.Module):
    def __init__(self):
        super(VGG19BN, self).__init__()
        self.vgg19_bn = torchvision.models.vgg19_bn(num_classes=10)
        self.features = self.vgg19_bn.features
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.BatchNorm1d(128),
            nn.Dropout(),
            nn.Linear(128, 10),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG19BN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    bestAcc = 0.0
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    epochNums = 200

    for epoch in range(epochNums):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        pbar = tqdm(enumerate(trainloader), total=len(trainloader))

        for i, data in pbar:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            pbar.set_description(f"Epoch [{epoch+1}/{epochNums}]")
            pbar.set_postfix(
                {
                    "Loss": running_loss / (i + 1),
                    "Accuracy": 100 * correct_train / total_train,
                }
            )

        trainLoss = running_loss / len(trainloader)
        train_accuracy = 100 * correct_train / total_train

        correctVal = 0
        totalVal = 0
        valLoss = 0.0

        model.eval()

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                valLoss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                totalVal += labels.size(0)
                correctVal += (predicted == labels).sum().item()

        valAcc = 100 * correctVal / totalVal
        valLoss /= len(testloader)

        print(
            f"Epoch [{epoch+1}/{epochNums}], "
            f"Training Loss: {trainLoss:.4f}, Training Accuracy: {train_accuracy:.2f}%, "
            f"Validation Loss: {valLoss:.4f}, Validation Accuracy: {valAcc:.2f}%"
        )

        if valAcc > bestAcc:
            bestAcc = valAcc
            torch.save(model.state_dict(), "vgg19_bn.pth")

        train_loss_list.append(trainLoss)
        val_loss_list.append(valLoss)
        train_acc_list.append(train_accuracy)
        val_acc_list.append(valAcc)

    epochs = range(1, int(epochNums) + 1)
    plt.figure()

    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_loss_list, color="blue", label="Train Loss")
    plt.plot(epochs, val_loss_list, color="orange", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_acc_list, color="blue", label="Train Acc")
    plt.plot(epochs, val_acc_list, color="orange", label="Validation Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("5-4.png")
    plt.show()
