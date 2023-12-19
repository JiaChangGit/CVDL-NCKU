import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from PIL import ImageFile
from tqdm import tqdm

# fix image file is truncated(warning)  not working!!  UserWarning: Truncated File Read   warnings.warn(str(msg))
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define data transformations
transform_with_erasing = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.RandomErasing(),
    ]
)

transform_without_erasing = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ]
)

# Load your datasets
train_data_dir = "C:\\Users\\louis\\Desktop\\cv\\Dataset_CvDl_Hw2\\training_dataset"
validation_data_dir = (
    "C:\\Users\\louis\\Desktop\\cv\\Dataset_CvDl_Hw2\\validation_dataset"
)

original_train_dataset = datasets.ImageFolder(
    root=train_data_dir, transform=transform_with_erasing
)

# Calculate the size for training and validation subsets
dataset_size = len(original_train_dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

# Create subsets for training and validation
train_dataset, val_dataset = random_split(
    original_train_dataset, [train_size, val_size]
)

# Create DataLoader for training and validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

if __name__ == "__main__":
    # Define ResNet50 models
    model_with_erasing = models.resnet50()
    model_without_erasing = models.resnet50()

    # Modify the last layer for your specific task (e.g., classification)
    num_classes = len(original_train_dataset.classes)  # num_classes = 2
    print("num_classes: " + str(num_classes))
    model_with_erasing.fc = nn.Linear(model_with_erasing.fc.in_features, num_classes)
    model_without_erasing.fc = nn.Linear(
        model_without_erasing.fc.in_features, num_classes
    )

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_with_erasing = optim.Adam(model_with_erasing.parameters(), lr=0.002)
    optimizer_without_erasing = optim.Adam(model_without_erasing.parameters(), lr=0.002)

    # Training loop
    num_epochs = 16
    train_accuracy_with_erasing = []
    train_accuracy_without_erasing = []
    val_accuracy_with_erasing = []
    val_accuracy_without_erasing = []

    for epoch in range(num_epochs):
        # Training with Random Erasing
        model_with_erasing.train()
        with tqdm(
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}, Training with Random Erasing",
        ) as pbar:
            for inputs, labels in train_loader:
                optimizer_with_erasing.zero_grad()
                outputs = model_with_erasing(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer_with_erasing.step()
                pbar.update(1)  # Update the progress bar

        # Validation with Random Erasing
        model_with_erasing.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model_with_erasing(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy_with_erasing = correct / total
        val_accuracy_with_erasing.append(accuracy_with_erasing)

        # Training without Random Erasing
        model_without_erasing.train()
        with tqdm(
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}, Training without Random Erasing",
        ) as pbar:
            for inputs, labels in train_loader:
                optimizer_without_erasing.zero_grad()
                outputs = model_without_erasing(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer_without_erasing.step()
                pbar.update(1)  # Update the progress bar

        # Validation without Random Erasing
        model_without_erasing.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model_without_erasing(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy_without_erasing = correct / total
        val_accuracy_without_erasing.append(accuracy_without_erasing)

        print(
            f"Epoch {epoch + 1}, Accuracy with Random Erasing: {accuracy_with_erasing:.4f}, Accuracy without Random Erasing: {accuracy_without_erasing:.4f}"
        )

    # Save the trained models
    torch.save(model_with_erasing.state_dict(), "model_with_erasing.pth")
    torch.save(model_without_erasing.state_dict(), "model_without_erasing.pth")

    # Plot and save the accuracy values
    bar_width = 0.35

    # Calculate average accuracy values
    avg_accuracy_with_erasing = sum(val_accuracy_with_erasing) / len(
        val_accuracy_with_erasing
    )
    avg_accuracy_without_erasing = sum(val_accuracy_without_erasing) / len(
        val_accuracy_without_erasing
    )

    plt.figure(figsize=(10, 5))

    # Plotting average accuracy values
    plt.bar(
        ["With Random Erasing", "Without Random Erasing"],
        [avg_accuracy_with_erasing, avg_accuracy_without_erasing],
        color=["blue", "orange"],
    )

    plt.xlabel("Data Augmentation")
    plt.ylabel("Average Validation Accuracy")
    plt.title("Average Validation Accuracy with and without Random Erasing")
    plt.savefig("Comparison.png")
    plt.show()
