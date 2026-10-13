import sys
import torch
from PIL import Image
from torchvision import transforms
from models.cnn_model import RoadDamageCNN

if len(sys.argv) != 2:
    raise SystemExit("Usage: python -m src.predict path/to/road_image.jpg")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

model = RoadDamageCNN().to(DEVICE)
model.load_state_dict(torch.load("outputs/best_model.pth", map_location=DEVICE))
model.eval()

image = Image.open(sys.argv[1]).convert("RGB")
x = tf(image).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    score = torch.sigmoid(model(x)).item()

print("Prediction:", "Potholes" if score >= 0.5 else "Normal road")
print(f"Pothole score: {score:.2%}")
