'''
 Copyright 2020 Xilinx Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''


'''
Common functions for simple PyTorch MNIST example
'''

'''
Author: Mark Harvey, Xilinx inc
'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import kagglehub
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


class CNN(nn.Module):
    def __init__(self, num_classes=29):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),        # 200x200 -> 100x100

            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),        # 100x100 -> 50x50

            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),        # 50x50 -> 25x25

            # Conv Block 4
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),        # 25x25 -> 12x12

            nn.Dropout(0.3),
            nn.Flatten()
        )

        # fully connected classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 12 * 12, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train(model, device, train_loader, optimizer, epoch):
    '''
    Train the model with tqdm progress bar
    '''
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        x = model(data)
        output = F.log_softmax(x, dim=1)   # ✅ dim=1 for class scores
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # update progress bar with loss
        pbar.set_postfix({"loss": loss.item()})


def test(model, device, test_loader):
    '''
    Evaluate the model with tqdm progress bar
    '''
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing", unit="batch")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.2f}%)\n")
    return acc


def compute_mean_std(dataset):
    # accumulate sum and sum of squares
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    mean = 0.0
    sq_mean = 0.0
    n_samples = 0
    for imgs, _ in loader:
        batch_samples = imgs.size(0)
        imgs = imgs.view(batch_samples, imgs.size(1), -1)  # [B, C, H*W]
        mean += imgs.mean(2).sum(0)
        sq_mean += (imgs ** 2).mean(2).sum(0)
        n_samples += batch_samples
    mean /= n_samples
    sq_mean /= n_samples
    std = (sq_mean - mean ** 2).sqrt()
    return mean, std


def get_asl_dataloader(batch_size = 256, test_size = 0.2):
    path = kagglehub.dataset_download("grassknoted/asl-alphabet")
    print("Path to dataset files:", path)

    data_root = os.path.join(path, "asl_alphabet_train/asl_alphabet_train")  # adjust if you see an extra nested folder

    full_dataset = datasets.ImageFolder(root=data_root, transform=transforms.ToTensor())
    print(f"Classes: {full_dataset.classes}")
    targets = [label for _, label in full_dataset.samples]

    train_idx, test_idx = train_test_split(
        range(len(full_dataset)),
        test_size=test_size,
        stratify=targets,
        random_state=42
    )
    img_size = 200
    train_dataset = Subset(full_dataset, train_idx)
    test_dataset = Subset(full_dataset, test_idx)
    mean, std = compute_mean_std(train_dataset)
    print(f"Mean: {mean.tolist()}\nstd: {std.tolist()}")
    
    train_transforms = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])

    train_dataset.dataset.transform = train_transforms
    test_dataset.dataset.transform = test_transforms

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader



