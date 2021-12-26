import torch

import torch.nn as nn
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.optim as optim
# TensorBoard
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchvision import models
# Load customized dataset
from dataset import DogCatDataset
from random_erasing import RandomErasing

# Setting TensorBoard
writer = SummaryWriter('./experiment')

def train(
    num_epoch, model, optimizer, criterion, 
    train_loader: DataLoader, 
    device:str, 
    random_earsing = False
    ):

    model.to(device)
    if random_earsing:
        re = RandomErasing()

    train_steps = 1
    best_acc =  0.0
    for epoch in range(num_epoch):
        loss_now = 0.0
        correct_now = 0

        losses = []
        model.train() ## important
        tqdm_loader = tqdm(enumerate(train_loader), total=len(train_loader))
        
        for batch_idx, (data, targets) in tqdm_loader:
            data = data.to(device=device)
            if random_earsing:
                data = re(data)
            targets = targets.to(device=device)
            outputs = model(data) # outputs

            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            losses.append(loss)

            # for tensorboard
            writer.add_scalar('training_loss', loss.item(), train_steps)
            train_steps += 1

            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)

            loss_now += loss.item() * data.size(0)
            correct_now += (preds == targets).sum().item()

            acc = round(correct_now / len(train_loader.dataset), 2) * 100

            
            if acc > best_acc:
                best_acc = acc
            writer.add_scalar('training_accuracy', best_acc, train_steps)
            tqdm_loader.set_description(
                f"Epoch {epoch+1}/{num_epoch}: process: {int((batch_idx / len(train_loader)) * 100)} "
                f"Accuracy: {acc} %"
            )
            tqdm_loader.set_postfix(loss=loss.data.item())

            torch.save(
                { 
                    'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(), 
                }, 
                '/models/cp_epoch_'+str(epoch)+'.pt',
                _use_new_zipfile_serialization=False
            )


def test(model, criterion, test_loader: DataLoader, device: str):
    model.eval()

    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for x, y in test_loader:            
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            _, preds = torch.max(output, 1)

            correct += (preds == y).sum().item()
            test_loss = criterion(output, y)
                                    
    test_loss /= len(test_loader.dataset)
    print(
        f"Average Loss: { test_loss} Accuracy: {correct}/{len(test_loader.dataset)}"
        f" {(correct / len(test_loader.dataset)) * 100}%"
    )
    return (correct / len(test_loader.dataset)) * 100


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageFolder("./train/")    

    train_data, test_data, train_label, test_label = train_test_split(
        dataset.imgs, 
        dataset.targets, 
        test_size=0.2, 
        random_state=3048
    )

    # use transform to transform the image to a specific formula
    trans = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])


    train_set = DogCatDataset(train_data, trans)
    test_set = DogCatDataset(test_data, trans)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True)


    # load homework required resnet50
    model = models.resnet50(pretrained=True)

    # for finetuning => set requires_grad = False
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train(5, model, optimizer, criterion, train_loader, device)

    test(model, criterion, test_loader, device)
    
    SAVE_PATH = './models/resNet50.model'
    torch.save(model.state_dict(), SAVE_PATH, _use_new_zipfile_serialization=False)



if __name__ == '__main__':
    main()