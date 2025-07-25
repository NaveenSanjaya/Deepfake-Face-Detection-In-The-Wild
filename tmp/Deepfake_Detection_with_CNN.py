# -*- coding: utf-8 -*-
"""GAN_Deepfake_Detection.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1CdurArA5R40DQiJJl5VLJheNyYELryzq
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import multiprocessing
from tqdm import tqdm
import os

from google.colab import drive
drive.mount('/content/drive')

'''import tarfile
import os
from tqdm import tqdm

def extract_tar_file(tar_path, extract_path):
    """
    Extracts a .tar file to the specified directory with a progress bar.

    :param tar_path: Path to the .tar file
    :param extract_path: Directory where the contents will be extracted
    """
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    with tarfile.open(tar_path, "r:*") as tar:
        members = tar.getmembers()
        total_members = len(members)

        with tqdm(total=total_members, desc="Extracting", ncols=80) as pbar:
            for member in members:
                tar.extract(member, path=extract_path)
                pbar.update(1)

    print(f"Extracted {tar_path} to {extract_path}")

# Example usage
tar_path = "/content/drive/MyDrive/Projects/IEEE - SP Cup/Dataset/Tar Files/train_fake.tar"
extract_path = "/content/drive/MyDrive/Projects/IEEE - SP Cup/Dataset/train/"
extract_tar_file(tar_path, extract_path)'''

data_dir = '/content/drive/MyDrive/Dataset'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/val'

def compute_mean_and_std(data_dir):
    """
    Compute per-channel mean and std of the dataset (to be used in transforms.Normalize())
    """
    cache_file = "mean_and_std.pt"
    if os.path.exists(cache_file):
        print(f"Reusing cached mean and std")
        d = torch.load(cache_file)
        return d["mean"], d["std"]

    ds = datasets.ImageFolder(
        data_dir, transform=transforms.Compose([transforms.ToTensor()])
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=1, num_workers=multiprocessing.cpu_count()
    )

    mean = torch.zeros(3)
    var = torch.zeros(3)
    npix = 0

    for images, _ in tqdm(dl, total=len(ds), desc="Computing mean and std", ncols=80):
        images = images.view(3, -1)
        npix += images.size(1)
        mean += images.mean(1)
        var += images.var(1, unbiased=False)

    mean /= len(ds)
    std = torch.sqrt(var / len(ds))

    # Cache results so we don't need to redo the computation
    torch.save({"mean": mean, "std": std}, cache_file)

    return mean, std

mean, std = compute_mean_and_std(train_dir)
print(f"Mean: {mean}")
print(f"Standard Deviation: {std}")

"""## Step 3: Dataset Preparation"""

# Define transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(10),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.3996, 0.3194, 0.3223], [0.2321, 0.1766, 0.1816])
])

valid_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.3996, 0.3194, 0.3223], [0.2321, 0.1766, 0.1816])

])


# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, num_workers=2)

"""## Step 5: Visualize Train Images"""

import matplotlib.pyplot as plt

# Map labels to "Real" or "Fake"
label_map = {0: "Fake", 1: "Real"}

# Get a batch of images from the training loader
data_iter = iter(trainloader)
images, labels = next(data_iter)

# Plot the first 5 images with their labels
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i in range(5):
    ax = axes[i]
    ax.imshow(images[i].permute(1, 2, 0))  # Convert from [C, H, W] to [H, W, C]
    ax.set_title(label_map[labels[i].item()], fontsize=12, color='black')
    ax.axis('off')
plt.tight_layout()
plt.show()

"""## Step 4: Define the Model"""

# Load pretrained resnet50
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = models.resnet50(weights=None)
print(model)

num_features = model.fc.in_features  # Get the number of features from the current fc layer
model.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 2), # Output layer for binary classification (Fake/Real)
    nn.Sigmoid(),
)

"""for p in model.fc.parameters():
    p.requires_grad = True"""

print(model)

"""## Step 6: Define Loss and Optimizer"""

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = criterion.to(device)

"""## Step 7: Training the Framework"""

def save_checkpoint(epoch, model, optimizer, loss, file_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved at epoch {epoch}.")

# Function to load the model and optimizer states
def load_checkpoint(file_path, model, optimizer):
    if os.path.exists(file_path):
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded from epoch {epoch}.")
        return epoch, loss
    else:
        print("Checkpoint file not found.")
        return 0, None

num_epochs = 100
save_every = 10
checkpoint_path = "model_checkpoint.pth"

start_epoch, _ = load_checkpoint(checkpoint_path, model, optimizer)

# Training loop
for epoch in range(start_epoch, num_epochs):
    model.train()
    train_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(trainloader):

        inputs, labels = inputs.to(device), labels.to(device)

        labels = labels.long()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(trainloader)}], Loss: {loss.item():.4f}')

    if (epoch + 1) % save_every == 0:
        save_checkpoint(epoch + 1, model, optimizer, epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {train_loss/len(trainloader):.4f}')

"""## Step 8: Save the Model"""

# Save the trained model
save_checkpoint(num_epochs, model, optimizer, epoch_loss)

"""## Step 9: Validating the Model"""

from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# Switch to evaluation mode
model.eval()

# Initialize variables to track metrics
correct = 0
total = 0
all_labels = []
all_predictions = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Disable gradient computation for validation
with torch.no_grad():
    for inputs, labels in validloader:
        # Move inputs and labels to the device (CPU or GPU)
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        # Get the predicted class index (0 or 1)
        _, predicted = torch.max(outputs, 1)  # Get the index of the max value along dim=1

        total += labels.size(0)
        correct += (predicted == labels).sum().item()  # Compare with true labels

        # Store all labels and predictions for further metrics calculation
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# Calculate accuracy as the percentage of correct predictions
accuracy = 100 * correct / total

# Calculate the confusion matrix
conf_matrix = confusion_matrix(all_labels, all_predictions)
tn, fp, fn, tp = conf_matrix.ravel()

# Print the metrics
print(f'Validation Accuracy: {accuracy:.2f}%')
print(f'True Negatives (Real identified as Real): {tn}')
print(f'False Positives (Real identified as Fake): {fp}')
print(f'False Negatives (Fake identified as Real): {fn}')
print(f'True Positives (Fake identified as Fake): {tp}')
print('Confusion Matrix:')
print(conf_matrix)