import os
from matplotlib import pyplot as plt
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from src.AWA.awaEncoder import AWAEmbeddingExtractor

class AwA2Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the AwA2 dataset files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        self.classes_file = os.path.join(root_dir, 'classes.txt')
        self.predicates_file = os.path.join(root_dir, 'predicate-matrix-binary.txt')
        self.images_dir = os.path.join(root_dir, 'JPEGImages')
        
        # 1. Load class names to create a class-to-index mapping
        self.class_to_idx = {}
        with open(self.classes_file, 'r') as f:
            for line in f:
                # AwA2 classes.txt format: "1 antelope"
                parts = line.strip().split()
                if len(parts) == 2:
                    idx, class_name = int(parts[0]) - 1, parts[1] # 0-indexed
                    self.class_to_idx[class_name] = idx

        # 2. Load the binary predicate matrix (attributes)
        # Rows represent classes, columns represent the 85 attributes
        predicate_matrix = np.loadtxt(self.predicates_file, dtype=np.float32)
        self.attributes_matrix = torch.tensor(predicate_matrix)

        # 3. Collect all image paths and their corresponding class indices
        self.image_paths = []
        self.targets = []
        self.labels = []
        
        for class_name in os.listdir(self.images_dir):
            class_dir = os.path.join(self.images_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            if class_name in self.class_to_idx:
                class_idx = self.class_to_idx[class_name]
                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.jpg') or img_name.endswith('.jpeg'):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        target = torch.zeros(len(self.class_to_idx), dtype=torch.float32)
                        target[class_idx] = 1.0
                        self.targets.append(target)
                        self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations (e.g., resizing, converting to tensor)
        if self.transform:
            image = self.transform(image)
            
        # Get label and corresponding attributes
        label = self.labels[idx]
        target = self.targets[idx]
        attributes = self.attributes_matrix[label]
        
        return image, attributes, target

def get_awa2_dataloader(root_dir, batch_size=32, shuffle=True, num_workers=4):
    """
    Creates and returns a DataLoader for the AwA2 dataset.
    """
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = AwA2Dataset(root_dir=root_dir, transform=transform)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
    )
    
    return dataloader


def get_awa2_dataloaders(root_dir, batch_size=32, shuffle=True, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = AwA2Dataset(root_dir=root_dir, transform=transform)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    realTrainSize = int(0.8 * train_size)
    valSize = train_size-realTrainSize
    test_size = total_size - train_size 

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [realTrainSize, valSize, test_size], 
        generator=generator
    )

    # 3. Create separate DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,  # Always shuffle training data
        num_workers=4,
        drop_last=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False, # No need to shuffle validation data
        num_workers=4,
        drop_last=False,
        persistent_workers=True
    )

    testLoader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, # No need to shuffle test data
        num_workers=4,
        drop_last=False,
        persistent_workers=True
    )
    return train_loader, val_loader, testLoader

from tqdm import tqdm # For a nice progress bar

def extract_and_save_embeddings(dataloader, embedder, save_path):
    embedder.eval() # Ensure the model is in evaluation mode
    embedder.to("mps")
    
    all_embeddings = []
    all_labels = []
    
    # We don't need gradients since we are just extracting
    with torch.no_grad():
        for images, _, labels in tqdm(dataloader, desc="Extracting"):
            images = images.to("mps")
            
            # Forward pass through the frozen embedder
            embeddings = embedder(images) 
            
            # Move back to CPU and store
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)
            
    # Concatenate lists into single massive tensors
    final_embeddings = torch.cat(all_embeddings, dim=0)
    final_labels = torch.cat(all_labels, dim=0)
    
    # Save to a PyTorch file
    torch.save({
        'embeddings': final_embeddings,
        'labels': final_labels
    }, save_path)
    
    print(f"Saved embeddings shape: {final_embeddings.shape}")


class AwA2EmbeddingDataset(Dataset):
    def __init__(self, embeddings_file, root_dir):
        """
        Args:
            embeddings_file: Path to the .pt file created in Step 1.
            root_dir: The AwA2 root directory (to load the predicate matrix).
        """
        # 1. Load the entire dataset of embeddings and labels into RAM
        data = torch.load(embeddings_file)
        self.embeddings = data['embeddings']
        self.labels = data['labels']
        self.indices = self.labels.to(torch.int64).argmax(dim=-1)
        
        # 2. Load the binary predicate matrix (attributes) exactly as before
        predicates_file = os.path.join(root_dir, 'predicate-matrix-binary.txt')
        predicate_matrix = np.loadtxt(predicates_file, dtype=np.float32)
        self.attributes_matrix = torch.tensor(predicate_matrix)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        indexLabel = self.indices[idx]
        attributes = self.attributes_matrix[indexLabel]
        
        return embedding, attributes, label


def getAWA2EmbeddingDataloaders(embeddings_file, root_dir, batch_size=32, shuffle=True, num_workers=4):
    dataset = AwA2EmbeddingDataset(embeddings_file, root_dir)
    
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    realTrainSize = int(0.8 * train_size)
    valSize = train_size-realTrainSize
    test_size = total_size - train_size 

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [realTrainSize, valSize, test_size], 
        generator=generator
    )

    # 3. Create separate DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,  # Always shuffle training data
        num_workers=4,
        drop_last=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, # No need to shuffle validation data
        num_workers=4,
        drop_last=False,
        persistent_workers=True
    )

    testLoader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, # No need to shuffle test data
        num_workers=4,
        drop_last=False,
        persistent_workers=True
    )
    #batch = next(iter(train_loader))
    #print("Sample batch shapes:")
    #print("Embeddings:", batch[0].shape)
    #print("Attributes:", batch[1].shape)
    #print("Labels:", batch[2].shape)
    return train_loader, val_loader, testLoader

# --- Example Usage ---
if __name__ == "__main__":
    # Replace with the actual path to your AwA2 dataset folder
    root = "/Users/alecvandeuren/Thesis/src/data/raw/AWA"
    dataLoader = get_awa2_dataloader(root_dir=root, batch_size=32, shuffle=True, num_workers=4)
    extract_and_save_embeddings(dataLoader, AWAEmbeddingExtractor(), "/Users/alecvandeuren/Thesis/src/data/raw/AWA/awa_embeddings.pt")
