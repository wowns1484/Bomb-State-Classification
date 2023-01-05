from torch.utils.data import Dataset
import glob
import random
import cv2
import albumentations as A

class BombDataset(Dataset):
    def __init__(self, dataset_path, mode, transforms, split_rate) -> None:
        super().__init__()
        
        self.images = []
        self.transforms = transforms
        root_path = sorted(glob.glob(dataset_path + "/*"))
        # "A": 0, "B": 1, "C": 2, "F": 3
        labels = [0, 1, 2, 3]
        
        for dir_path, label in zip(root_path, labels):
            images = [[image, label] for image in glob.glob(dir_path + "/*.png")]
            random.shuffle(images)
            
            if mode == "train":
                self.images += images[:int(len(images)*split_rate)]
            elif mode == "val":
                self.images += images[int(len(images)*split_rate):]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image, label = self.images[index]
        
        # label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=4)
        
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transforms != None:
            image = self.transforms(image=image)['image']
        
        return image, label