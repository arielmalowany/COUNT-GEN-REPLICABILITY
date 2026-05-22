"""
Fine-tune ResNet-18 on CelebA-HQ for 13-attribute classification.

Usage:
    python train_celeba_hq_classifier.py \
        --image_dir /path/to/celeba_hq/images/ \
        --attr_path /path/to/CelebAMask-HQ-attribute-anno.txt \
        --output_dir ./checkpoints \
        --epochs 10 \
        --batch_size 64 \
        --lr 1e-4 \
        --img_size 384
"""

import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ──────────────────────────────────────────────
# 1. Configuration
# ──────────────────────────────────────────────

# The 13 attributes used by Att-GAN (same order as Att-GAN training)
ATTGAN_ATTRS = [
    'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair',
    'Bushy_Eyebrows', 'Eyeglasses', 'Male', 'Mouth_Slightly_Open',
    'Mustache', 'No_Beard', 'Pale_Skin', 'Young'
]


# ──────────────────────────────────────────────
# 2. Dataset
# ──────────────────────────────────────────────

class CelebAHQDataset(Dataset):
    """
    CelebA-HQ dataset for multi-label attribute classification.
    
    Expects:
        - image_dir: folder with images named like 0.jpg, 1.jpg, ..., 29999.jpg
        - attr_path: path to CelebAMask-HQ-attribute-anno.txt
          (first line: number of images, second line: attribute names,
           remaining lines: filename followed by -1/1 labels)
    """
    def __init__(self, image_dir, attr_path, selected_attrs, transform=None, split='train'):
        self.image_dir = image_dir
        self.transform = transform
        self.selected_attrs = selected_attrs

        # Parse the attribute file
        # CelebAMask-HQ-attribute-anno.txt format:
        # Line 1: number of images (30000)
        # Line 2: attribute names separated by spaces
        # Line 3+: filename  attr1 attr2 ... attr40
        with open(attr_path, 'r') as f:
            lines = f.readlines()

        n_images = int(lines[0].strip())
        all_attr_names = lines[1].strip().split()

        # Get indices of selected attributes
        self.attr_indices = [all_attr_names.index(a) for a in selected_attrs]

        # Parse image entries
        self.filenames = []
        self.labels = []
        for line in lines[2:]:
            parts = line.strip().split()
            filename = parts[0]
            attrs = [int(x) for x in parts[1:]]
            selected = [attrs[i] for i in self.attr_indices]
            # Convert from {-1, 1} to {0, 1}
            selected = [(x + 1) // 2 for x in selected]
            self.filenames.append(filename)
            self.labels.append(selected)

        self.labels = np.array(self.labels, dtype=np.float32)

        # Split: 24000 train, 3000 val, 3000 test
        n_total = len(self.filenames)
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.1)

        if split == 'train':
            self.filenames = self.filenames[:n_train]
            self.labels = self.labels[:n_train]
        elif split == 'val':
            self.filenames = self.filenames[n_train:n_train + n_val]
            self.labels = self.labels[n_train:n_train + n_val]
        elif split == 'test':
            self.filenames = self.filenames[n_train + n_val:]
            self.labels = self.labels[n_train + n_val:]

        print(f"[{split}] Loaded {len(self.filenames)} images, {len(selected_attrs)} attributes")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.filenames[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


# ──────────────────────────────────────────────
# 3. Model
# ──────────────────────────────────────────────

class AttrClassifier(nn.Module):
    """
    ResNet-18 fine-tuned for multi-label attribute classification.
    
    Forward returns logits (before sigmoid).
    Use .predict() for sigmoid outputs.
    Use .predict_continuous() for logits as continuous attribute representation.
    """
    def __init__(self, num_attrs=13, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, num_attrs),
        )

    def forward(self, x):
        """Returns raw logits."""
        return self.backbone(x)

    def predict(self, x):
        """Returns probabilities (after sigmoid)."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def predict_continuous(self, x):
        """Returns logits as continuous attribute representation for f_att."""
        return self.forward(x)


# ──────────────────────────────────────────────
# 4. Training
# ──────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.numel()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # Per-attribute tracking
    attr_correct = None
    attr_total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.numel()

        # Per-attribute accuracy
        batch_correct = (preds == labels).float().sum(dim=0)
        if attr_correct is None:
            attr_correct = batch_correct
        else:
            attr_correct += batch_correct
        attr_total += images.size(0)

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / total
    per_attr_acc = attr_correct / attr_total
    return avg_loss, accuracy, per_attr_acc


# ──────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train CelebA-HQ attribute classifier')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Path to CelebA-HQ images folder')
    parser.add_argument('--attr_path', type=str, required=True,
                        help='Path to CelebAMask-HQ-attribute-anno.txt')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Where to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=384,
                        help='Image resolution (default 384 to match Att-GAN)')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, cuda, cuda:0, etc.')
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Datasets
    train_dataset = CelebAHQDataset(
        args.image_dir, args.attr_path, ATTGAN_ATTRS,
        transform=train_transform, split='train'
    )
    val_dataset = CelebAHQDataset(
        args.image_dir, args.attr_path, ATTGAN_ATTRS,
        transform=val_transform, split='val'
    )
    test_dataset = CelebAHQDataset(
        args.image_dir, args.attr_path, ATTGAN_ATTRS,
        transform=val_transform, split='test'
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model
    model = AttrClassifier(num_attrs=len(ATTGAN_ATTRS), pretrained=True).to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # Training loop
    best_val_acc = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_per_attr = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train  loss: {train_loss:.4f}  acc: {train_acc:.4f}")
        print(f"  Val    loss: {val_loss:.4f}  acc: {val_acc:.4f}")
        print(f"  Per-attribute accuracy:")
        for attr, acc in zip(ATTGAN_ATTRS, val_per_attr):
            print(f"    {attr:25s}: {acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.output_dir, 'best_attr_classifier.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'selected_attrs': ATTGAN_ATTRS,
            }, save_path)
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")

    # Final test evaluation
    print("\n" + "=" * 60)
    print("Final evaluation on test set")
    print("=" * 60)

    # Load best model
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_attr_classifier.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_per_attr = evaluate(model, test_loader, criterion, device)
    print(f"  Test   loss: {test_loss:.4f}  acc: {test_acc:.4f}")
    print(f"  Per-attribute accuracy:")
    for attr, acc in zip(ATTGAN_ATTRS, test_per_attr):
        print(f"    {attr:25s}: {acc:.4f}")


# ──────────────────────────────────────────────
# 6. Inference utility (import from other scripts)
# ──────────────────────────────────────────────

def load_classifier(checkpoint_path, device='cpu'):
    """
    Load trained classifier for inference.
    
    Returns model and transform.
    
    Usage:
        model, transform = load_classifier('checkpoints/best_attr_classifier.pth')
        img = Image.open('face.jpg').convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        
        # Continuous logits for f_att (what you need for the optimization)
        logits = model.predict_continuous(img_tensor)
        
        # Binary predictions
        probs = model.predict(img_tensor)
        preds = (probs > 0.5).int()
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_attrs = len(checkpoint['selected_attrs'])

    model = AttrClassifier(num_attrs=num_attrs, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return model, transform


if __name__ == '__main__':
    main()
