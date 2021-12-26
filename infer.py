import torchvision.transforms as transforms 
import torch
from torchvision import models
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

def infer(filepath, model, device):
    model.eval()
    img_array = Image.open(filepath).convert("RGB")
    data_transforms=transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    img = data_transforms(img_array).unsqueeze(dim=0) 
    load = DataLoader(img)
    
    for x in load:
        x=x.to(device)
        pred = model(x)
        _, preds = torch.max(pred, 1)
        print(f"class : {preds}")
        if preds[0] == 1: 
            print(f"Dog")
            return "Dog"
        else: 
            print(f"Cat")
            return "Cat"

def infer_by_test_idx(idx: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load homework required resnet50
    model = models.resnet50(pretrained=True)    

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.to(device)

    PATH = './models/resNet50.model'
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    
    img_path = f'test/{idx}.jpg'

    return infer(img_path, model, device), img_path