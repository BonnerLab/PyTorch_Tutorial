from torch.utils.data import Dataset
from torchvision.transforms import functional as tr
import os
from PIL import Image


# Dataset classes are used to load images and do pre-processing on them so that they can be fed to your model
class CustomDataset(Dataset):

    def __init__(self, data_dir, resolution):
        self.resolution = resolution    # (height, width) tuple you want to resize your images to

        data = os.listdir(data_dir)                         # Get a list of every filename in the data directory
        data = [d for d in data if '.jpg' in d]             # Filter out any files that aren't jpg images
        data = [os.path.join(data_dir, d) for d in data]    # Turn the filenames into file paths
        self.data = data

    # Called whenever you use len(dataset)
    def __len__(self):
        return len(self.data)

    # Called whenever you use dataset[item] to index into it
    def __getitem__(self, item):
        image = self.data[item]                     # Get the path to the image file
        image = Image.open(image).convert('RGB')    # Read the image file as an RGB image
        image = self.transform(image)               # Transform the image to a format the model can run
        return image

    def transform(self, image):
        image = tr.resize(image, size=self.resolution)
        image = tr.to_tensor(image)     # Converts an RGB image object into a PyTorch tensor object (0.0<=values<=1.0)
        image = tr.normalize(image, mean=[0.485, 0.456, 0.406],     # ImageNet models are trained on data
                                    std=[0.229, 0.224, 0.225])      # normalized to a different range
        return image
