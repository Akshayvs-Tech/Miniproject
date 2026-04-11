# =====================================================
# SCRIPT 2 — train_lstm.py
#
# Trains an LSTM classifier on the keypoint sequences
# collected by collect_keypoints.py
#
# Input:
#   keypoints.npy
#   labels.npy
#   classes.txt
#
# Output:
#   action_lstm.pth   — trained LSTM model
#   classes.txt       — class names (copied to output)
#
# Usage:
#   python train_lstm.py
# =====================================================

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

# =====================================================
# Config
# =====================================================

SEQ_LEN     = 30      # must match collect_keypoints.py
INPUT_SIZE  = 99      # 33 landmarks x 3 (x, y, z)
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.5
BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 1e-3
VAL_SPLIT   = 0.2


# =====================================================
# Dataset
# =====================================================

class KeypointDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)   # (N, SEQ_LEN, 99)
        self.y = torch.LongTensor(y)    # (N,)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =====================================================
# LSTM Model
# =====================================================

class ActionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0
        )
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out     = self.dropout(out[:, -1, :])   # take last timestep
        return self.classifier(out)


# =====================================================
# Training
# =====================================================

def train():
    # Load data
    X = np.load("keypoints.npy")   # (N, SEQ_LEN, 99)
    y = np.load("labels.npy")      # (N,)

    with open("classes.txt") as f:
        classes = [l.strip() for l in f.readlines()]

    num_classes = len(classes)
    print(f"✅ Loaded {len(X)} sequences, {num_classes} classes: {classes}")

    # Split train/val
    dataset    = KeypointDataset(X, y)
    val_size   = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"✅ Train: {train_size}  Val: {val_size}")

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Device: {device}")

    model = ActionLSTM(
        input_size  = INPUT_SIZE,
        hidden_size = HIDDEN_SIZE,
        num_layers  = NUM_LAYERS,
        num_classes = num_classes,
        dropout     = DROPOUT
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total   = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss    += loss.item()
            _, predicted   = outputs.max(1)
            train_correct += predicted.eq(y_batch).sum().item()
            train_total   += y_batch.size(0)

        train_acc = 100.0 * train_correct / train_total

        # Validate
        model.eval()
        val_correct = 0
        val_total   = 0
        val_loss    = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                loss    = criterion(outputs, y_batch)
                val_loss     += loss.item()
                _, predicted  = outputs.max(1)
                val_correct  += predicted.eq(y_batch).sum().item()
                val_total    += y_batch.size(0)

        val_acc = 100.0 * val_correct / val_total
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:02d}/{EPOCHS}  "
              f"Train Loss: {train_loss/len(train_loader):.4f}  "
              f"Train Acc: {train_acc:.1f}%  "
              f"Val Acc: {val_acc:.1f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes":          classes,
                "input_size":       INPUT_SIZE,
                "hidden_size":      HIDDEN_SIZE,
                "num_layers":       NUM_LAYERS,
                "num_classes":      num_classes,
            }, "action_lstm.pth")
            print(f"  💾 Best model saved (val acc: {val_acc:.1f}%)")

    print(f"\n✅ Training complete — best val accuracy: {best_val_acc:.1f}%")
    print(f"✅ Model saved to action_lstm.pth")


if __name__ == "__main__":
    train()