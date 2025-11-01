import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import requests
from io import BytesIO

class FlowerDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(self.df['label'].unique()))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if 'image' in row and isinstance(row['image'], dict):
            image_data = row['image']
            if 'bytes' in image_data:
                image = Image.open(BytesIO(image_data['bytes'])).convert('RGB')
            elif 'path' in image_data:
                image = Image.open(image_data['path']).convert('RGB')
            else:
                raise ValueError("Unknown image format")
        elif isinstance(row['image'], str):
            if row['image'].startswith('http'):
                response = requests.get(row['image'])
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                image = Image.open(row['image']).convert('RGB')
        else:
            image = row['image'].convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.label_to_idx[row['label']]
        return image, label

class FlowerClassifier(nn.Module):
    def __init__(self, num_classes=102, use_pretrained=True):
        super(FlowerClassifier, self).__init__()

        if use_pretrained:
            self.model = models.mobilenet_v2(pretrained=True)
            in_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(in_features, num_classes)
            )
        else:
            self.model = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes)
            )

    def forward(self, x):
        return self.model(x)

class FlowerTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.class_names = []

    def load_dataset(self, dataset_url: str):
        print(f"Loading dataset from: {dataset_url}")
        df = pd.read_parquet(dataset_url)

        if 'label' not in df.columns:
            if 'category' in df.columns:
                df['label'] = df['category']
            elif 'class' in df.columns:
                df['label'] = df['class']
            else:
                raise ValueError("Dataset must have a 'label', 'category', or 'class' column")

        train_size = int(0.8 * len(df))
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:]

        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        train_dataset = FlowerDataset(train_df, transform=transform)
        val_dataset = FlowerDataset(val_df, transform=transform)

        self.class_names = [train_dataset.idx_to_label[i] for i in range(len(train_dataset.idx_to_label))]

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            num_workers=0
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=0
        )

        print(f"Dataset loaded: {len(train_dataset)} training, {len(val_dataset)} validation samples")
        print(f"Number of classes: {len(self.class_names)}")

        return len(self.class_names)

    def load_multiple_datasets(self, dataset_urls: List[str]):
        all_train_dfs = []
        all_val_dfs = []

        for url in dataset_urls:
            print(f"Loading dataset from: {url}")
            df = pd.read_parquet(url)

            if 'label' not in df.columns:
                if 'category' in df.columns:
                    df['label'] = df['category']
                elif 'class' in df.columns:
                    df['label'] = df['class']

            train_size = int(0.8 * len(df))
            all_train_dfs.append(df.iloc[:train_size])
            all_val_dfs.append(df.iloc[train_size:])

        train_df = pd.concat(all_train_dfs, ignore_index=True)
        val_df = pd.concat(all_val_dfs, ignore_index=True)

        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        train_dataset = FlowerDataset(train_df, transform=transform)
        val_dataset = FlowerDataset(val_df, transform=transform)

        self.class_names = [train_dataset.idx_to_label[i] for i in range(len(train_dataset.idx_to_label))]

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            num_workers=0
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=0
        )

        print(f"Combined dataset loaded: {len(train_dataset)} training, {len(val_dataset)} validation samples")
        print(f"Number of classes: {len(self.class_names)}")

        return len(self.class_names)

    def build_model(self, num_classes: int):
        use_pretrained = self.config.get('use_pretrained', True)
        self.model = FlowerClassifier(num_classes=num_classes, use_pretrained=use_pretrained)
        self.model = self.model.to(self.device)

        lr = self.config.get('learning_rate', 1e-4)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        print(f"Model built with {num_classes} classes on device: {self.device}")
        print(f"Using pretrained: {use_pretrained}")

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        return val_loss, val_acc

    def train(self, num_epochs: int):
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")

        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        best_val_acc = 0.0

        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('best_flower_model.pth')

        print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
        return history

    def save_model(self, filename: str):
        model_dir = Path('backend/models')
        model_dir.mkdir(exist_ok=True)

        save_path = model_dir / filename

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'config': self.config
        }, save_path)

        print(f"Model saved to {save_path}")

        with open(model_dir / 'class_names.json', 'w') as f:
            json.dump(self.class_names, f, indent=2)

    def load_model(self, filename: str):
        model_path = Path('backend/models') / filename
        checkpoint = torch.load(model_path, map_location=self.device)

        self.class_names = checkpoint['class_names']
        self.config = checkpoint['config']

        num_classes = len(self.class_names)
        self.build_model(num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Model loaded from {model_path}")

def train_from_config(config_path: str):
    with open(config_path, 'r') as f:
        config = json.load(f)

    trainer = FlowerTrainer(config)

    dataset_urls = config.get('dataset_urls', [])
    if len(dataset_urls) == 1:
        num_classes = trainer.load_dataset(dataset_urls[0])
    else:
        num_classes = trainer.load_multiple_datasets(dataset_urls)

    trainer.build_model(num_classes)

    num_epochs = config.get('num_epochs', 10)
    history = trainer.train(num_epochs)

    final_model_name = config.get('model_name', 'flower_model.pth')
    trainer.save_model(final_model_name)

    return history

if __name__ == '__main__':
    config = {
        'dataset_urls': [
            'hf://datasets/huggan/flowers-102-categories/data/train-00000-of-00001.parquet'
        ],
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 10,
        'use_pretrained': True,
        'model_name': 'flower_model.pth'
    }

    config_dir = Path('backend/training_configs')
    config_dir.mkdir(exist_ok=True)

    with open(config_dir / 'default_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("Starting training with default configuration...")
    history = train_from_config('backend/training_configs/default_config.json')
    print("\nTraining history saved!")
