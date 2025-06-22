import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model architecture and weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model.load_state_dict(torch.load("model.pth", map_location=device))
model = model.to(device)
model.eval()

# Define preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Prediction function
def predict(image):
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    return "Ghibli Image" if predicted.item() == 1 else " Original Image"

# Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Text(label="Prediction"),
    title="Ghibli Image Classifier",
    description="Upload an image to find out whether it's Ghibli-style or a real photo!"
)

if __name__ == "__main__":
    interface.launch(share=True)
