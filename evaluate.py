import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from models.cnn_model import RoadDamageCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data/road_damage_data"
MODEL_PATH = "outputs/best_model.pth"

tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
ds = datasets.ImageFolder(DATA_DIR + "/test", transform=tf)
assert ds.class_to_idx == {"normal": 0, "potholes": 1}
dl = DataLoader(ds, batch_size=16, shuffle=False)

model = RoadDamageCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
actual, predicted = [], []

with torch.no_grad():
    for images, labels in dl:
        outputs = model(images.to(DEVICE))
        preds = (torch.sigmoid(outputs) >= 0.5).long().cpu().numpy().ravel()
        predicted.extend(preds)
        actual.extend(labels.numpy())

results = {
    "Accuracy": accuracy_score(actual, predicted),
    "Precision": precision_score(actual, predicted, zero_division=0),
    "Recall": recall_score(actual, predicted, zero_division=0),
    "F1-score": f1_score(actual, predicted, zero_division=0)
}
for name, value in results.items():
    print(f"{name}: {value:.4f}")

print("\\nClassification report:")
print(classification_report(actual, predicted,
      labels=[0, 1], target_names=["Normal", "Potholes"], zero_division=0))
print("Confusion matrix:")
print(confusion_matrix(actual, predicted, labels=[0, 1]))

with open("outputs/test_results.txt", "w") as f:
    for name, value in results.items():
        f.write(f"{name}: {value:.4f}\\n")
    f.write("\\nConfusion matrix:\\n")
    f.write(str(confusion_matrix(actual, predicted, labels=[0, 1])))
