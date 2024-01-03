import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torch.optim import Adam
from PIL import Image, UnidentifiedImageError
import os
import warnings
import logging
from tqdm import tqdm  # Import tqdm for progress bars
from PIL import ImageFile

# fix image file is truncated(warning)  not working!!  UserWarning: Truncated File Read   warnings.warn(str(msg))
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Set random seed for reproducibility
torch.manual_seed(42)

# Use torch.device for GPU/CPU selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations
transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.RandomErasing(),
    ]
)

# Custom DataLoader to handle Truncated File Read warning
class CustomImageLoader(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.loader = self.default_loader
        self.samples = self._make_dataset()

    def _make_dataset(self):
        extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        samples = []
        class_to_idx = {}
        for idx, target_class in enumerate(sorted(os.listdir(self.root))):
            class_to_idx[target_class] = idx
            class_path = os.path.join(self.root, target_class)
            if not os.path.isdir(class_path):
                continue
            for root, _, fnames in sorted(os.walk(class_path)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if self.is_valid_file(path, extensions):
                        item = path, idx
                        samples.append(item)
        self.class_to_idx = class_to_idx
        return samples

    def is_valid_file(self, path, extensions):
        return path.lower().endswith(extensions)

    def default_loader(self, path):
        try:
            return Image.open(path).convert('RGB')
        except (FileNotFoundError, UnidentifiedImageError) as e:
            print(f"Error loading image: {e}")
            # Handle the error by returning a placeholder image
            return Image.new('RGB', (224, 224), (0, 0, 0))

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)

# Create ImageFolder datasets for training and validation
train_dataset = CustomImageLoader(root='./Dataset_CvDl_Hw2/training_dataset', transform=transform)
val_dataset = CustomImageLoader(root='./Dataset_CvDl_Hw2/validation_dataset', transform=transform)

# Define data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Instantiate the model, move it to the device
class ResNet50BinaryClassifier(nn.Module):
    def __init__(self):
        super(ResNet50BinaryClassifier, self).__init__()

        # Suppress deprecation warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

        # Load the pre-trained ResNet50 model
        resnet50_model = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet50_model.children())[:-1])

        # Add a new fully connected layer with 1 output node and a Sigmoid activation function
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x

# Use torch.device for moving the model to GPU/CPU
model = ResNet50BinaryClassifier().to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Use torch.optim.lr_scheduler for learning rate scheduling
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop with tqdm progress bars
num_epochs = 10

if __name__ == '__main__':
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        if epoch > 10:
            optimizer = Adam(model.parameters(), lr=0.0001)

        # Use tqdm for progress bar
        tqdm_train_loader = tqdm(enumerate(train_loader, 1), total=len(train_loader),
                                 desc=f'Epoch {epoch + 1}/{num_epochs}')
        for i, (inputs, labels) in tqdm_train_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels.view(-1, 1)).sum().item()

            # Update tqdm progress bar description
            train_accuracy = (correct / total) * 100
            tqdm_train_loader.set_postfix({'Loss': loss.item(), 'Train Accuracy': train_accuracy})

        # Average loss and accuracy for the epoch
        avg_loss = total_loss / len(train_loader)
        train_accuracy = (correct / total) * 100
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Average Train Loss: {avg_loss:.4f}, '
                    f'Average Train Accuracy: {train_accuracy:.2f}%')

        # Validation loop with tqdm progress bar
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            # Use tqdm for progress bar
            tqdm_val_loader = tqdm(enumerate(val_loader, 1), total=len(val_loader),
                                   desc=f'Validation Epoch {epoch + 1}/{num_epochs}')
            for j, (inputs_val, labels_val) in tqdm_val_loader:
                inputs_val, labels_val = inputs_val.to(device), labels_val.float().to(device)
                outputs_val = model(inputs_val)
                loss_val = criterion(outputs_val, labels_val.view(-1, 1))
                val_loss += loss_val.item()

                predicted_val = (outputs_val > 0.5).float()
                total_val += labels_val.size(0)
                correct_val += (predicted_val == labels_val.view(-1, 1)).sum().item()

                # Update tqdm progress bar description
                val_accuracy = (correct_val / total_val) * 100
                tqdm_val_loader.set_postfix({'Validation Loss': loss_val.item(), 'Validation Accuracy': val_accuracy})

        # Average validation loss and accuracy for the epoch
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = (correct_val / total_val) * 100
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Average Validation Loss: {avg_val_loss:.4f}, '
                    f'Average Validation Accuracy: {val_accuracy:.2f}%')

    # Save the trained model with metadata
    metadata = {
        'architecture': 'ResNet50BinaryClassifier',
        'transformations': 'RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ToTensor, RandomErasing',
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'learning_rate': 0.001,
        'final_learning_rate': 0.0001
    }
    torch.save({'metadata': metadata, 'model_state_dict': model.state_dict()}, './resnet50_RE.pth')
