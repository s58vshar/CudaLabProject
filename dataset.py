import os
import torch
from torch.utils.data import Dataset, default_collate
from PIL import Image
import random
from torchvision import transforms
import numpy as np

class ConvertToRGB(object):
    def __call__(self, img):
        # Check if image has 4 channels (RGBA)
        if img.shape[0] == 4:
            # Discard the alpha channel
            img = img[:3, :, :]
        return img
    
class DepthImageTransform:
    def __call__(self, image):

        depth = (image[0] + image[1]*256.0 + image[2]*65536.0)/(16777215.0)
        log_depth = torch.log(depth + 1).unsqueeze(0)
        
        return log_depth
    
def gaussian_kernel(center, dim, std):
    """Generate a Gaussian kernel."""
    n, m = torch.meshgrid(torch.arange(dim[0]), torch.arange(dim[1]), indexing='ij')
    n, m = n.float(), m.float()
    center_n, center_m = center
    std_n, std_m = std
    kernel = torch.exp(-0.5 * (((n - center_n) / std_n) ** 2 + ((m - center_m) / std_m) ** 2))
    kernel = kernel / torch.sum(kernel)
    return kernel

def custom_collate(batch):
    filtered_batch = [item for item in batch if item[0] is not None and item[1] is not None]
        
    if len(filtered_batch) == 0:
        raise RuntimeError("No valid samples in the batch after filtering out None items.")
    return default_collate(filtered_batch)

class SegmentationDataset(Dataset):
    def __init__(self, data_dir, towns=['00', '01', '03'], mode=0, corrupted = False):
        self.data_dir = data_dir
        self.towns = towns
        self.corrupted = corrupted
        self.mode = mode

        if(self.mode == 1) :
            self.image_transform = transforms.Compose([
        transforms.Resize((256, 512)),  # Resize images to a fixed size
        transforms.ToTensor(),
        ConvertToRGB(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        ])

        else :
            self.image_transform = transforms.Compose([
        transforms.Resize((256, 512)),  # Resize images to a fixed size
        transforms.ToTensor(),
        ConvertToRGB(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                  
])
        self.depth_transform = transforms.Compose([
        transforms.Resize((256, 512)),  # Resize images to a fixed size
        transforms.ToTensor(), 
        ConvertToRGB(),
        DepthImageTransform()          
])
        self.segmentation_transform = transforms.Compose([
        transforms.Resize((256, 512)),  # Resize images to a fixed size
        transforms.ToTensor(), 
        ConvertToRGB()          
])

        self.sequences = []

        for town in self.towns:
            town_dir = os.path.join(self.data_dir, f'Town{town}')
            sequence_dirs = [os.path.join(town_dir, d) for d in os.listdir(town_dir) if os.path.isdir(os.path.join(town_dir, d))]

            for sequence_dir in sequence_dirs:
                image_files = sorted([f for f in os.listdir(sequence_dir) if f.endswith('.png') and 'img_' in f])
                segmentation_files = sorted([f for f in os.listdir(sequence_dir) if f.endswith('.png') and 'segmentation_' in f])
                depth_files = sorted([f for f in os.listdir(sequence_dir) if f.endswith('.png') and 'depth_' in f])

                sequence_data = []
                for image_file, segmentation_file, depth_file in zip(image_files, segmentation_files, depth_files):
                    image_path = os.path.join(sequence_dir, image_file)
                    segmentation_path = os.path.join(sequence_dir, segmentation_file)
                    depth_path = os.path.join(sequence_dir, depth_file)

                    sequence_data.append((image_path, segmentation_path, depth_path))

                self.sequences.append(sequence_data)

    def __len__(self):
        return len(self.sequences)
    
    
    def corrupt_frames(self, frames):
        _,_, h, w = frames.shape
        # Clutter
        clutter_mask = torch.zeros((1, 1, h, w)).to(frames.device)
        for _ in range(random.randint(1, 5)):
            center = (random.randint(0, h - 1), random.randint(0, w - 1))
            std = (random.uniform(15, 50), random.uniform(15, 50))
            clutter_mask += gaussian_kernel(center=center, dim=(h, w), std=std)
        clutter_mask = clutter_mask.clamp(0, 1)
        frames = frames * (1 - clutter_mask) + (clutter_mask * torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(frames.device))

        # Illumination changes and noise should be adapted if needed.

        return frames.clamp_(0.0, 1.0)

    def __getitem__(self, idx):
        sequence_data = self.sequences[idx]

        start_index = random.randint(0, 14)
        selected_sequence_data = sequence_data[start_index:start_index+6]

        sequence_images = []
        sequence_segmentations = []
        sequence_depths = []

        for image_path, segmentation_path, depth_path in selected_sequence_data:
            image = Image.open(image_path)
            segmentation = Image.open(segmentation_path)
            depth = Image.open(depth_path)

            if self.image_transform:
                image = self.image_transform(image)

            if self.segmentation_transform:
                segmentation = self.segmentation_transform(segmentation)
            
            if self.depth_transform:
                depth = self.depth_transform(depth)

            sequence_images.append(image)
            sequence_segmentations.append(segmentation)
            sequence_depths.append(depth)
        
        if len(sequence_images) != len(sequence_segmentations) or len(sequence_images) != len(sequence_depths) or len(sequence_images) == 0 :
            print(f"Skipping sequence due to inconsistent number of images, segmentations, or depth images.")
            return None, None, None

        sequence_images = torch.stack(sequence_images)
        sequence_segmentations = torch.stack(sequence_segmentations)
        sequence_depths = torch.stack(sequence_depths)

        if self.corrupted and random.random() <=0.1:
                sequence_images = self.corrupt_frames(sequence_images)

        return sequence_images, sequence_segmentations, sequence_depths
