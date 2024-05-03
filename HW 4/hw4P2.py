# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:36:29 2024

@author: jebowman
"""
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import Dataset
import pandas as pd
import h5py
import os
from PIL import Image  
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import random


# Define a function to display both correct and incorrect images side by side
def display_correct_incorrect_images(correct_images, correct_labels, incorrect_images, incorrect_labels, predicted_labels):
    num_samples = min(len(correct_images), len(incorrect_images))
    plt.figure(figsize=(12, 6 * num_samples))
    for i in range(num_samples):
        plt.subplot(num_samples, 2, 2*i + 1)
        plt.imshow(correct_images[i].permute(1, 2, 0))
        plt.title(f"Correct - True: {correct_labels[i]}, Predicted: {predicted_labels[i]}")
        plt.axis('off')
        
        plt.subplot(num_samples, 2, 2*i + 2)
        plt.imshow(incorrect_images[i].permute(1, 2, 0))
        plt.title(f"Incorrect - True: {incorrect_labels[i]}, Predicted: {predicted_labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Define your test dataset class
class CustomTestDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')  
        label = self.data.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

# Define the transformations for your test dataset
test_transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the root directory and CSV file for the test dataset
test_data_dir = r"C:\Users\jebowman\OneDrive - Iowa State University\Documents\Courses\Graduate\ME 592\Homework #4\images"
test_csv_file = r"C:\Users\jebowman\OneDrive - Iowa State University\Documents\Courses\Graduate\ME 592\Homework #4\labels.csv"  

# Create the test dataset
test_dataset = CustomTestDataset(csv_file=test_csv_file, root_dir=test_data_dir, transform=test_transform)

# Create the DataLoader for the test dataset
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the pretrained ConvNeXt model
model = models.convnext_tiny(pretrained=True)

# Modify the last linear layer in the classifier block
in_features = model.classifier[-1].in_features
num_classes = 9  
model.classifier[-1] = nn.Linear(in_features, num_classes)

# Move the model to the appropriate device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()

# Evaluate the model on the test set
model.eval()
test_loss = 0
correct = 0
total = 0
predicted_labels = []
true_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Compute the average test loss
avg_test_loss = test_loss / len(test_loader)
print(f"Average Test Loss: {avg_test_loss:.4f}")

# Compute the test accuracy
test_accuracy = 100.0 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Compute the confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(conf_matrix)

# Display a sample of 9 correctly classified images
correct_images = []
correct_labels = []
for i in range(len(predicted_labels)):
    if len(correct_images) >= 9:
        break
    if predicted_labels[i] == true_labels[i]:
        correct_images.append(test_dataset[i][0])
        correct_labels.append(true_labels[i])  # Directly append the label

# Display a sample of 9 incorrectly classified images
incorrect_images = []
incorrect_labels = []
for i in range(len(predicted_labels)):
    if len(incorrect_images) >= 9:
        break
    if predicted_labels[i] != true_labels[i]:
        incorrect_images.append(test_dataset[i][0])
        incorrect_labels.append(true_labels[i])  # Directly append the label

# Display both correctly and incorrectly classified images
display_correct_incorrect_images(correct_images, correct_labels, incorrect_images, incorrect_labels, predicted_labels)









