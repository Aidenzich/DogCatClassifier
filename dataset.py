from torch.utils.data import DataLoader, Dataset
from PIL import Image

class DogCatDataset(Dataset):
    def __init__(self, dataset, transform=None) -> None:
        super().__init__()
        self.transform = transform
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image = Image.open(self.dataset[item][0])

        label = self.dataset[item][1]

        if self.transform:
            image = self.transform(image)
        return image, label
        