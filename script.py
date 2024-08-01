import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

class Plant_Disease_Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.network = models.resnet34(pretrained=True)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 38)

    def forward(self, xb):
        out = self.network(xb)
        return out


transform = transforms.Compose(
    [transforms.Resize(size=128), transforms.ToTensor()])

num_classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 
'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 
'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 
'Peach___healthy','Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']


model = Plant_Disease_Model()
model.load_state_dict(torch.load('plantDisease-resnet34.pth', map_location=torch.device('cpu')))
model.eval()


def predict_image_path(img_path):
    img = Image.open(img_path)
    tensor = transform(img)
    xb = tensor.unsqueeze(0)
    with torch.no_grad():
        yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return num_classes[preds[0].item()]

def test_images_in_folder(folder_path):
    predictions = {}
    for img_name in os.listdir(folder_path):
        if img_name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(folder_path, img_name)
            prediction = predict_image_path(img_path)
            predictions[img_name] = prediction
    return predictions

# Path to the Test folder
test_folder_path = './Test'

# Get predictions for all images in the Test folder
predictions = test_images_in_folder(test_folder_path)

# Print out the predictions
for img_name, prediction in predictions.items():
    print(f'{img_name}: {prediction}')