import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from models.cnn_model import RoadDamageCNN

DATA_DIR = "data/road_damage_data"
OUT_DIR = "outputs"
IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUT_DIR, exist_ok=True)

train_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
eval_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

train_ds = datasets.ImageFolder(DATA_DIR + "/train", transform=train_tf)
val_ds = datasets.ImageFolder(DATA_DIR + "/val", transform=eval_tf)
assert train_ds.class_to_idx == {"normal": 0, "potholes": 1}
assert train_ds.class_to_idx == val_ds.class_to_idx

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

model = RoadDamageCNN().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
best_loss = float("inf")

print("Device:", DEVICE)
print("Class mapping:", train_ds.class_to_idx)

for epoch in range(EPOCHS):
    model.train()
    total_loss, preds, labels_all = 0.0, [], []

    for images, labels in train_dl:
        images, labels = images.to(DEVICE), labels.float().unsqueeze(1).to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds.extend((torch.sigmoid(outputs) >= 0.5).int().detach().cpu().numpy().ravel())
        labels_all.extend(labels.detach().cpu().numpy().ravel())

    train_loss = total_loss / len(train_ds)
    train_acc = accuracy_score(labels_all, preds)

    model.eval()
    val_loss, val_preds, val_labels = 0.0, [], []
    with torch.no_grad():
        for images, labels in val_dl:
            images, labels = images.to(DEVICE), labels.float().unsqueeze(1).to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            val_preds.extend((torch.sigmoid(outputs) >= 0.5).int().cpu().numpy().ravel())
            val_labels.extend(labels.cpu().numpy().ravel())

    val_loss /= len(val_ds)
    val_acc = accuracy_score(val_labels, val_preds)
    print(f"Epoch {epoch+1}/{EPOCHS} | train loss {train_loss:.4f} | "
          f"train accuracy {train_acc:.4f} | val loss {val_loss:.4f} | "
          f"val accuracy {val_acc:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), OUT_DIR + "/best_model.pth")

print("Best model saved in outputs/best_model.pth")
