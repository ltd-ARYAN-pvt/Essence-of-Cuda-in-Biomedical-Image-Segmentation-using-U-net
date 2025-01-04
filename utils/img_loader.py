import os
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as T

# Dataset for Train/Validation
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, transform=None):
        """
        Args:
            image_paths (list): List of image file paths.
            mask_paths (list, optional): List of mask file paths. None for test data.
            transform (callable, optional): Transform to be applied on both image and mask.
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.mask_paths:
            mask_path = self.mask_paths[idx]
            mask = Image.open(mask_path).convert("L")  # Grayscale mask
        else:
            mask = None  # Test dataset case

        if self.transform:
            if mask:
                # Apply same transform to both image and mask
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]
            else:
                # Apply transform only to the image
                image = self.transform(image=image)["image"]

        return image, mask


# Dataset for Test (masks generated from CSV)
class TestDatasetWithCSV(Dataset):
    def __init__(self, image_dir, csv_path, transform=None, size=(256, 256)):
        """
        Args:
            image_dir (str): Directory containing test images.
            csv_path (str): Path to the CSV file with mask coordinates.
            transform (callable, optional): Transform to be applied on the image.
            size (tuple, optional): Resize dimensions for the image and mask.
        """
        self.image_dir = image_dir
        self.csv_path = csv_path
        self.transform = transform
        self.size = size

        # Load CSV
        self.annotations = pd.read_csv(csv_path)
        self.image_filenames = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Create mask from coordinates in CSV
        mask = Image.new("L", self.size)
        draw = ImageDraw.Draw(mask)
        annotations = self.annotations[self.annotations["filename"] == img_name]

        for _, row in annotations.iterrows():
            # Draw polygons from CSV (adjust logic for your specific CSV structure)
            coords = eval(row["coordinates"])  # Assuming coordinates are stored as a list of tuples
            draw.polygon(coords, outline=255, fill=255)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask


# Transform Function
class ImageMaskTransform:
    def __init__(self, size=(256, 256)):
        self.size = size

    def __call__(self, image, mask=None):
        # Resize
        image = T.Resize(self.size)(image)
        if mask:
            mask = T.Resize(self.size)(mask)

        # Convert to tensor
        image = T.ToTensor()(image)
        if mask:
            mask = T.ToTensor()(mask)

        # Normalize image
        image = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image)
        return {"image": image, "mask": mask}