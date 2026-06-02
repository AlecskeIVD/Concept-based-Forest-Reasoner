from collections import defaultdict
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from src.CUB.cubEncoder import CUBEmbeddingExtractor

SELECTED_CONCEPTS = [
    1,
    4,
    6,
    7,
    10,
    14,
    15,
    20,
    21,
    23,
    25,
    29,
    30,
    35,
    36,
    38,
    40,
    44,
    45,
    50,
    51,
    53,
    54,
    56,
    57,
    59,
    63,
    64,
    69,
    70,
    72,
    75,
    80,
    84,
    90,
    91,
    93,
    99,
    101,
    106,
    110,
    111,
    116,
    117,
    119,
    125,
    126,
    131,
    132,
    134,
    145,
    149,
    151,
    152,
    153,
    157,
    158,
    163,
    164,
    168,
    172,
    178,
    179,
    181,
    183,
    187,
    188,
    193,
    194,
    196,
    198,
    202,
    203,
    208,
    209,
    211,
    212,
    213,
    218,
    220,
    221,
    225,
    235,
    236,
    238,
    239,
    240,
    242,
    243,
    244,
    249,
    253,
    254,
    259,
    260,
    262,
    268,
    274,
    277,
    283,
    289,
    292,
    293,
    294,
    298,
    299,
    304,
    305,
    308,
    309,
    310,
    311,
]

# Names of all CUB attributes
CONCEPT_SEMANTICS = [
    "has_bill_shape::curved_(up_or_down)",
    "has_bill_shape::dagger",
    "has_bill_shape::hooked",
    "has_bill_shape::needle",
    "has_bill_shape::hooked_seabird",
    "has_bill_shape::spatulate",
    "has_bill_shape::all-purpose",
    "has_bill_shape::cone",
    "has_bill_shape::specialized",
    "has_wing_color::blue",
    "has_wing_color::brown",
    "has_wing_color::iridescent",
    "has_wing_color::purple",
    "has_wing_color::rufous",
    "has_wing_color::grey",
    "has_wing_color::yellow",
    "has_wing_color::olive",
    "has_wing_color::green",
    "has_wing_color::pink",
    "has_wing_color::orange",
    "has_wing_color::black",
    "has_wing_color::white",
    "has_wing_color::red",
    "has_wing_color::buff",
    "has_upperparts_color::blue",
    "has_upperparts_color::brown",
    "has_upperparts_color::iridescent",
    "has_upperparts_color::purple",
    "has_upperparts_color::rufous",
    "has_upperparts_color::grey",
    "has_upperparts_color::yellow",
    "has_upperparts_color::olive",
    "has_upperparts_color::green",
    "has_upperparts_color::pink",
    "has_upperparts_color::orange",
    "has_upperparts_color::black",
    "has_upperparts_color::white",
    "has_upperparts_color::red",
    "has_upperparts_color::buff",
    "has_underparts_color::blue",
    "has_underparts_color::brown",
    "has_underparts_color::iridescent",
    "has_underparts_color::purple",
    "has_underparts_color::rufous",
    "has_underparts_color::grey",
    "has_underparts_color::yellow",
    "has_underparts_color::olive",
    "has_underparts_color::green",
    "has_underparts_color::pink",
    "has_underparts_color::orange",
    "has_underparts_color::black",
    "has_underparts_color::white",
    "has_underparts_color::red",
    "has_underparts_color::buff",
    "has_breast_pattern::solid",
    "has_breast_pattern::spotted",
    "has_breast_pattern::striped",
    "has_breast_pattern::multi-colored",
    "has_back_color::blue",
    "has_back_color::brown",
    "has_back_color::iridescent",
    "has_back_color::purple",
    "has_back_color::rufous",
    "has_back_color::grey",
    "has_back_color::yellow",
    "has_back_color::olive",
    "has_back_color::green",
    "has_back_color::pink",
    "has_back_color::orange",
    "has_back_color::black",
    "has_back_color::white",
    "has_back_color::red",
    "has_back_color::buff",
    "has_tail_shape::forked_tail",
    "has_tail_shape::rounded_tail",
    "has_tail_shape::notched_tail",
    "has_tail_shape::fan-shaped_tail",
    "has_tail_shape::pointed_tail",
    "has_tail_shape::squared_tail",
    "has_upper_tail_color::blue",
    "has_upper_tail_color::brown",
    "has_upper_tail_color::iridescent",
    "has_upper_tail_color::purple",
    "has_upper_tail_color::rufous",
    "has_upper_tail_color::grey",
    "has_upper_tail_color::yellow",
    "has_upper_tail_color::olive",
    "has_upper_tail_color::green",
    "has_upper_tail_color::pink",
    "has_upper_tail_color::orange",
    "has_upper_tail_color::black",
    "has_upper_tail_color::white",
    "has_upper_tail_color::red",
    "has_upper_tail_color::buff",
    "has_head_pattern::spotted",
    "has_head_pattern::malar",
    "has_head_pattern::crested",
    "has_head_pattern::masked",
    "has_head_pattern::unique_pattern",
    "has_head_pattern::eyebrow",
    "has_head_pattern::eyering",
    "has_head_pattern::plain",
    "has_head_pattern::eyeline",
    "has_head_pattern::striped",
    "has_head_pattern::capped",
    "has_breast_color::blue",
    "has_breast_color::brown",
    "has_breast_color::iridescent",
    "has_breast_color::purple",
    "has_breast_color::rufous",
    "has_breast_color::grey",
    "has_breast_color::yellow",
    "has_breast_color::olive",
    "has_breast_color::green",
    "has_breast_color::pink",
    "has_breast_color::orange",
    "has_breast_color::black",
    "has_breast_color::white",
    "has_breast_color::red",
    "has_breast_color::buff",
    "has_throat_color::blue",
    "has_throat_color::brown",
    "has_throat_color::iridescent",
    "has_throat_color::purple",
    "has_throat_color::rufous",
    "has_throat_color::grey",
    "has_throat_color::yellow",
    "has_throat_color::olive",
    "has_throat_color::green",
    "has_throat_color::pink",
    "has_throat_color::orange",
    "has_throat_color::black",
    "has_throat_color::white",
    "has_throat_color::red",
    "has_throat_color::buff",
    "has_eye_color::blue",
    "has_eye_color::brown",
    "has_eye_color::purple",
    "has_eye_color::rufous",
    "has_eye_color::grey",
    "has_eye_color::yellow",
    "has_eye_color::olive",
    "has_eye_color::green",
    "has_eye_color::pink",
    "has_eye_color::orange",
    "has_eye_color::black",
    "has_eye_color::white",
    "has_eye_color::red",
    "has_eye_color::buff",
    "has_bill_length::about_the_same_as_head",
    "has_bill_length::longer_than_head",
    "has_bill_length::shorter_than_head",
    "has_forehead_color::blue",
    "has_forehead_color::brown",
    "has_forehead_color::iridescent",
    "has_forehead_color::purple",
    "has_forehead_color::rufous",
    "has_forehead_color::grey",
    "has_forehead_color::yellow",
    "has_forehead_color::olive",
    "has_forehead_color::green",
    "has_forehead_color::pink",
    "has_forehead_color::orange",
    "has_forehead_color::black",
    "has_forehead_color::white",
    "has_forehead_color::red",
    "has_forehead_color::buff",
    "has_under_tail_color::blue",
    "has_under_tail_color::brown",
    "has_under_tail_color::iridescent",
    "has_under_tail_color::purple",
    "has_under_tail_color::rufous",
    "has_under_tail_color::grey",
    "has_under_tail_color::yellow",
    "has_under_tail_color::olive",
    "has_under_tail_color::green",
    "has_under_tail_color::pink",
    "has_under_tail_color::orange",
    "has_under_tail_color::black",
    "has_under_tail_color::white",
    "has_under_tail_color::red",
    "has_under_tail_color::buff",
    "has_nape_color::blue",
    "has_nape_color::brown",
    "has_nape_color::iridescent",
    "has_nape_color::purple",
    "has_nape_color::rufous",
    "has_nape_color::grey",
    "has_nape_color::yellow",
    "has_nape_color::olive",
    "has_nape_color::green",
    "has_nape_color::pink",
    "has_nape_color::orange",
    "has_nape_color::black",
    "has_nape_color::white",
    "has_nape_color::red",
    "has_nape_color::buff",
    "has_belly_color::blue",
    "has_belly_color::brown",
    "has_belly_color::iridescent",
    "has_belly_color::purple",
    "has_belly_color::rufous",
    "has_belly_color::grey",
    "has_belly_color::yellow",
    "has_belly_color::olive",
    "has_belly_color::green",
    "has_belly_color::pink",
    "has_belly_color::orange",
    "has_belly_color::black",
    "has_belly_color::white",
    "has_belly_color::red",
    "has_belly_color::buff",
    "has_wing_shape::rounded-wings",
    "has_wing_shape::pointed-wings",
    "has_wing_shape::broad-wings",
    "has_wing_shape::tapered-wings",
    "has_wing_shape::long-wings",
    "has_size::large_(16_-_32_in)",
    "has_size::small_(5_-_9_in)",
    "has_size::very_large_(32_-_72_in)",
    "has_size::medium_(9_-_16_in)",
    "has_size::very_small_(3_-_5_in)",
    "has_shape::upright-perching_water-like",
    "has_shape::chicken-like-marsh",
    "has_shape::long-legged-like",
    "has_shape::duck-like",
    "has_shape::owl-like",
    "has_shape::gull-like",
    "has_shape::hummingbird-like",
    "has_shape::pigeon-like",
    "has_shape::tree-clinging-like",
    "has_shape::hawk-like",
    "has_shape::sandpiper-like",
    "has_shape::upland-ground-like",
    "has_shape::swallow-like",
    "has_shape::perching-like",
    "has_back_pattern::solid",
    "has_back_pattern::spotted",
    "has_back_pattern::striped",
    "has_back_pattern::multi-colored",
    "has_tail_pattern::solid",
    "has_tail_pattern::spotted",
    "has_tail_pattern::striped",
    "has_tail_pattern::multi-colored",
    "has_belly_pattern::solid",
    "has_belly_pattern::spotted",
    "has_belly_pattern::striped",
    "has_belly_pattern::multi-colored",
    "has_primary_color::blue",
    "has_primary_color::brown",
    "has_primary_color::iridescent",
    "has_primary_color::purple",
    "has_primary_color::rufous",
    "has_primary_color::grey",
    "has_primary_color::yellow",
    "has_primary_color::olive",
    "has_primary_color::green",
    "has_primary_color::pink",
    "has_primary_color::orange",
    "has_primary_color::black",
    "has_primary_color::white",
    "has_primary_color::red",
    "has_primary_color::buff",
    "has_leg_color::blue",
    "has_leg_color::brown",
    "has_leg_color::iridescent",
    "has_leg_color::purple",
    "has_leg_color::rufous",
    "has_leg_color::grey",
    "has_leg_color::yellow",
    "has_leg_color::olive",
    "has_leg_color::green",
    "has_leg_color::pink",
    "has_leg_color::orange",
    "has_leg_color::black",
    "has_leg_color::white",
    "has_leg_color::red",
    "has_leg_color::buff",
    "has_bill_color::blue",
    "has_bill_color::brown",
    "has_bill_color::iridescent",
    "has_bill_color::purple",
    "has_bill_color::rufous",
    "has_bill_color::grey",
    "has_bill_color::yellow",
    "has_bill_color::olive",
    "has_bill_color::green",
    "has_bill_color::pink",
    "has_bill_color::orange",
    "has_bill_color::black",
    "has_bill_color::white",
    "has_bill_color::red",
    "has_bill_color::buff",
    "has_crown_color::blue",
    "has_crown_color::brown",
    "has_crown_color::iridescent",
    "has_crown_color::purple",
    "has_crown_color::rufous",
    "has_crown_color::grey",
    "has_crown_color::yellow",
    "has_crown_color::olive",
    "has_crown_color::green",
    "has_crown_color::pink",
    "has_crown_color::orange",
    "has_crown_color::black",
    "has_crown_color::white",
    "has_crown_color::red",
    "has_crown_color::buff",
    "has_wing_pattern::solid",
    "has_wing_pattern::spotted",
    "has_wing_pattern::striped",
    "has_wing_pattern::multi-colored",
]

# SELECTED_CONCEPTS = SELECTED_CONCEPTS[:20]

CONCEPT_GROUP_MAP = defaultdict(list)
for i, concept_name in enumerate(list(
    np.array(CONCEPT_SEMANTICS)[SELECTED_CONCEPTS]
)):
    group = concept_name[:concept_name.find("::")]
    CONCEPT_GROUP_MAP[group].append(i)


class BirdDataset(Dataset):
    def __init__(self, root_dir, concept_map, selected_indices, imagesFile, splitFile, transform=None, train=True):
        """
        Args:
            root_dir (string): Directory with images.
            concept_map (dict): Mapping {filename: torch.Tensor(312)}
            selected_indices (list): List of attribute IDs (1-based) to keep.
            transform (callable, optional): Image transforms.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.concept_map = concept_map
        
        # Convert 1-based indices to 0-based for slicing
        self.selected_indices = [i - 1 for i in selected_indices]

        idToSplit = {}
        with open(splitFile, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_id = int(parts[0])
                    is_train = int(parts[1]) == 1
                    idToSplit[img_id] = is_train
        
        self.image_paths = []
        self.image_labels = []
        self.classes = set()

        # Build file list
        with open(imagesFile, 'r') as f:
            for line in f:
                parts = line.strip().split()
                img_id = int(parts[0])
                relative_path = parts[1] # e.g. "001.Black_footed_Albatross/image_01.jpg"
                if idToSplit[img_id] == (1 if train else 0):
                    full_path = os.path.join(root_dir, relative_path)
                    self.image_paths.append(full_path)
                    self.classes.add(relative_path.split('/')[0])
                    self.image_labels.append(len(self.classes)-1)
        self.num_classes = len(self.classes)
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Load Image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        # 2. Process Concepts
        filename = os.path.basename(img_path)
        
        # Get the full 312-dim vector from our map
        # If filename is missing, return zeros (safety)
        full_vector = self.concept_map.get(filename, torch.zeros(312))
        
        # Filter to only the SELECTED_CONCEPTS
        concept_vector = full_vector[self.selected_indices]

        # 3. Process Label
        label_idx = self.image_labels[idx]
        # Note: Often cross-entropy expects the index, 
        # but if you need one-hot:
        label_one_hot = torch.zeros(self.num_classes)
        label_one_hot[label_idx] = 1.0

        return image, concept_vector, label_one_hot


def create_concept_map(images_file_path, attributes_file_path):
    """
    Creates a dictionary mapping filenames to a pytorch tensor of attributes.
    
    Args:
        images_file_path: Path to CUB_200_2011/images.txt
        attributes_file_path: Path to CUB_200_2011/attributes/image_attribute_labels.txt
    
    Returns:
        fullConcept_map: Dictionary { 'filename': torch.tensor([0, 1, ...]) }
    """
    
    image_id_to_name = {}
    print(f"Reading {images_file_path}...")
    with open(images_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                img_id = int(parts[0])
                # The file often includes folder path (e.g., 001.Black_footed.../Image.jpg)
                # We use os.path.basename to get just 'Black_Footed_Albatross_0001_796111.jpg'
                full_path = parts[1]
                filename = os.path.basename(full_path) 
                image_id_to_name[img_id] = filename

    num_attributes = 312
    image_attributes = {}

    for img_id in image_id_to_name.keys():
        image_attributes[img_id] = torch.zeros(num_attributes, dtype=torch.float32)

    # 3. Parse Attributes File
    # Format: <image_id> <attribute_id> <is_present> <certainty_id> <time>
    print(f"Reading {attributes_file_path}...")
    with open(attributes_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                img_id = int(parts[0])
                attr_id = int(parts[1])
                is_present = int(parts[2])
                
                # Update tensor if attribute is present
                # Note: attr_id is 1-based (1-312), so we subtract 1 for 0-based indexing
                if is_present == 1:
                    if img_id in image_attributes:
                        image_attributes[img_id][attr_id - 1] = 1

    # 4. Construct final dictionary
    fullConcept_map = {}
    for img_id, filename in image_id_to_name.items():
        fullConcept_map[filename] = image_attributes[img_id]

    return fullConcept_map

def createDataloaders(batchSize: int) -> tuple[DataLoader, DataLoader]:
    data_transforms = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    imagesTXTPath = "/Users/alecvandeuren/Thesis/src/data/raw/CUB/CUB_fixed/images.txt"
    attributesTXTPath = "/Users/alecvandeuren/Thesis/src/data/raw/CUB/CUB_fixed/image_attribute_labels.txt"
    
    # This still returns the full 312-dim map
    full_map = create_concept_map(imagesTXTPath, attributesTXTPath)

    train_dataset = BirdDataset(
        root_dir='/Users/alecvandeuren/Thesis/src/data/raw/CUB/CUB_BASE/CUB_200_2011/images',
        concept_map=full_map,
        selected_indices=SELECTED_CONCEPTS, # Pass the filter here
        imagesFile=imagesTXTPath,
        splitFile="/Users/alecvandeuren/Thesis/src/data/raw/CUB/CUB_BASE/CUB_200_2011/train_test_split.txt",
        transform=data_transforms,
        train = True
    )

    test_dataset = BirdDataset(
        root_dir='/Users/alecvandeuren/Thesis/src/data/raw/CUB/CUB_fixed/test',
        concept_map=full_map,
        selected_indices=SELECTED_CONCEPTS,
        imagesFile=imagesTXTPath,
        splitFile="/Users/alecvandeuren/Thesis/src/data/raw/CUB/CUB_BASE/CUB_200_2011/train_test_split.txt",
        transform=data_transforms,
        train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers=4, persistent_workers=True)

    test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False, num_workers=4, persistent_workers=True)


    return train_loader, test_loader


class EmbeddedDataset(Dataset):
    def __init__(self, root_dir, concept_map, selected_indices, imagesFile, splitFile, encoder, transform=None, train=True):
        """
        Args:
            root_dir (string): Directory with images.
            concept_map (dict): Mapping {filename: torch.Tensor(312)}
            selected_indices (list): List of attribute IDs (1-based) to keep.
            transform (callable, optional): Image transforms.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.concept_map = concept_map
        
        # Convert 1-based indices to 0-based for slicing
        self.selected_indices = [i - 1 for i in selected_indices]
        self.encoder = encoder
        self.encoder.eval()
        self.cache = {}  # Stores {index: embedding_tensor}

        idToSplit = {}
        with open(splitFile, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_id = int(parts[0])
                    is_train = int(parts[1]) == 1
                    idToSplit[img_id] = is_train
        
        self.image_paths = []
        self.image_labels = []
        self.classes = set()

        # Build file list
        with open(imagesFile, 'r') as f:
            for line in f:
                parts = line.strip().split()
                img_id = int(parts[0])
                relative_path = parts[1] # e.g. "001.Black_footed_Albatross/image_01.jpg"
                if idToSplit[img_id] == (1 if is_train else 0):
                    full_path = os.path.join(root_dir, relative_path)
                    self.image_paths.append(full_path)
                    self.classes.add(relative_path.split('/')[0])
                    self.image_labels.append(len(self.classes)-1)
        self.num_classes = len(self.classes)
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Load Image
        img_path = self.image_paths[idx]
        if idx in self.cache:
            image = self.cache[idx]
        else:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            # pass through encoder
            with torch.no_grad():
                image = self.encoder(image.unsqueeze(0)).squeeze(0)
            
        # 2. Process Concepts
        filename = os.path.basename(img_path)
        
        # Get the full 312-dim vector from our map
        # If filename is missing, return zeros (safety)
        full_vector = self.concept_map.get(filename, torch.zeros(312))
        
        # Filter to only the SELECTED_CONCEPTS
        concept_vector = full_vector[self.selected_indices]

        # 3. Process Label
        label_idx = self.image_labels[idx]
        # Note: Often cross-entropy expects the index, 
        # but if you need one-hot:
        label_one_hot = torch.zeros(self.num_classes)
        label_one_hot[label_idx] = 1.0

        self.cache[idx] = image

        return image, concept_vector, label_one_hot

def createEmbeddedDataloaders(batchSize: int) -> tuple[DataLoader, DataLoader]:
    data_transforms = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    imagesTXTPath = "/Users/alecvandeuren/Thesis/src/data/raw/CUB/CUB_fixed/images.txt"
    attributesTXTPath = "/Users/alecvandeuren/Thesis/src/data/raw/CUB/CUB_fixed/image_attribute_labels.txt"
    
    # This still returns the full 312-dim map
    full_map = create_concept_map(imagesTXTPath, attributesTXTPath)

    train_dataset = EmbeddedDataset(
        root_dir='/Users/alecvandeuren/Thesis/src/data/raw/CUB/CUB_BASE/CUB_200_2011/images',
        concept_map=full_map,
        selected_indices=SELECTED_CONCEPTS, # Pass the filter here
        imagesFile=imagesTXTPath,
        splitFile="/Users/alecvandeuren/Thesis/src/data/raw/CUB/CUB_BASE/CUB_200_2011/train_test_split.txt",
        encoder=CUBEncoder(),
        transform=data_transforms,
        train = True
    )

    test_dataset = EmbeddedDataset(
        root_dir='/Users/alecvandeuren/Thesis/src/data/raw/CUB/CUB_fixed/test',
        concept_map=full_map,
        selected_indices=SELECTED_CONCEPTS,
        imagesFile=imagesTXTPath,
        splitFile="/Users/alecvandeuren/Thesis/src/data/raw/CUB/CUB_BASE/CUB_200_2011/train_test_split.txt",
        encoder=CUBEncoder(),
        transform=data_transforms,
        train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers=4, persistent_workers=True)

    test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False, num_workers=4, persistent_workers=True)


    return train_loader, test_loader


from tqdm import tqdm

def extract_and_save_embeddings(images_file, root_dir, output_file="cub_embeddings.pt", device="mps"):
    """
    Runs all CUB images through the encoder and saves them to a .pt file.
    """
    encoder = get_resnet18_embedder().to(device)
    encoder.eval()
    
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    embeddings_dict = {}
    
    print("Extracting embeddings...")
    with open(images_file, 'r') as f:
        lines = f.readlines()
        
    with torch.no_grad():
        for line in tqdm(lines):
            parts = line.strip().split()
            img_id = int(parts[0])
            relative_path = parts[1]
            full_path = os.path.join(root_dir, relative_path)
            
            if os.path.exists(full_path):
                # Load and transform image
                image = Image.open(full_path).convert('RGB')
                image_tensor = data_transforms(image).unsqueeze(0).to(device)
                
                # Get embedding
                embedding = encoder(image_tensor).squeeze(0).cpu() # Move back to CPU for storage
                
                # Store using the relative path as the key
                embeddings_dict[relative_path] = embedding

    # Save the dictionary to disk
    torch.save(embeddings_dict, output_file)
    print(f"Saved {len(embeddings_dict)} embeddings to {output_file}")


class PrecomputedEmbeddingDataset(Dataset):
    def __init__(self, embeddings_file, concept_map, selected_indices, imagesFile, splitFile, train=True, allowed_classes=None):
        """
        Args:
            embeddings_file (string): Path to the saved .pt embeddings dictionary.
            concept_map (dict): Mapping {filename: torch.Tensor(312)}
            selected_indices (list): List of attribute IDs (1-based) to keep.
            imagesFile (string): Path to images.txt
            splitFile (string): Path to train_test_split.txt
            train (bool): True for train split, False for test split.
        """
        self.concept_map = concept_map
        self.selected_indices = [i - 1 for i in selected_indices]
        
        # Load the precomputed embeddings into memory
        print(f"Loading embeddings from {embeddings_file}...")
        self.embeddings_dict = torch.load(embeddings_file)
        
        idToSplit = {}
        with open(splitFile, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_id = int(parts[0])
                    # Fix: use the dataset's `train` boolean
                    is_train_img = int(parts[1]) == 1 
                    idToSplit[img_id] = is_train_img
        
        self.image_keys = []
        self.image_labels = []
        self.classes = set()
        if allowed_classes is not None:
            self.allowed_classes = set(allowed_classes)
        else:
            all_classes = []
            with open(imagesFile, 'r') as f:
                for line in f:
                    class_name = line.strip().split()[1].split('/')[0]
                    if class_name not in all_classes:
                        all_classes.append(class_name)
            self.allowed_classes = set(all_classes)
            
        self.num_classes = len(self.allowed_classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(list(self.allowed_classes)))}

        # Build file list
        with open(imagesFile, 'r') as f:
            for line in f:
                parts = line.strip().split()
                img_id = int(parts[0])
                relative_path = parts[1]
                class_name = relative_path.split('/')[0]
                if class_name not in self.allowed_classes:
                    continue
                
                if idToSplit[img_id] == train:
                    # We store the relative path as the key to look up the embedding
                    self.image_keys.append(relative_path)
                    #self.classes.add(relative_path.split('/')[0])
                    self.image_labels.append(self.class_to_idx[class_name])
                    
        print(f"Loaded {len(self.image_keys)} samples across {self.num_classes} classes for {'Train' if train else 'Test'}.")

    def __len__(self):
        return len(self.image_keys)

    def __getitem__(self, idx):
        relative_path = self.image_keys[idx]
        
        embedding = self.embeddings_dict[relative_path]
            
        filename = os.path.basename(relative_path)
        full_vector = self.concept_map.get(filename, torch.zeros(312))
        concept_vector = full_vector[self.selected_indices]

        label_idx = self.image_labels[idx]
        label_one_hot = torch.zeros(self.num_classes)
        label_one_hot[label_idx] = 1.0

        return embedding, concept_vector, label_one_hot


def createPrecomputedDataloaders(batchSize: int, embeddings_file: str, num_random_classes: int = None) -> tuple[DataLoader, DataLoader, DataLoader]:
    imagesTXTPath = "/Users/alecvandeuren/Thesis/src/data/raw/CUB/CUB_fixed/images.txt"
    attributesTXTPath = "/Users/alecvandeuren/Thesis/src/data/raw/CUB/CUB_fixed/image_attribute_labels.txt"
    splitFile = "/Users/alecvandeuren/Thesis/src/data/raw/CUB/CUB_BASE/CUB_200_2011/train_test_split.txt"
    
    full_map = create_concept_map(imagesTXTPath, attributesTXTPath)
    allowed_classes = None
    if num_random_classes is not None:
        all_classes = []
        with open(imagesTXTPath, 'r') as f:
            for line in f:
                class_name = line.strip().split()[1].split('/')[0]
                if class_name not in all_classes:
                    all_classes.append(class_name)
        
        allowed_classes = random.sample(all_classes, num_random_classes)
        print(f"Randomly selected {num_random_classes} classes out of {len(all_classes)}.")

    train_dataset = PrecomputedEmbeddingDataset(
        embeddings_file=embeddings_file,
        concept_map=full_map,
        selected_indices=SELECTED_CONCEPTS,
        imagesFile=imagesTXTPath,
        splitFile=splitFile,
        train=True,
        allowed_classes=allowed_classes
    )

    trainSize = int(0.8 * len(train_dataset))
    valSize = len(train_dataset) - trainSize

    train_dataset, val_dataset = random_split(
        train_dataset, 
        [trainSize, valSize], 
    )

    test_dataset = PrecomputedEmbeddingDataset(
        embeddings_file=embeddings_file,
        concept_map=full_map,
        selected_indices=SELECTED_CONCEPTS,
        imagesFile=imagesTXTPath,
        splitFile=splitFile,
        train=False,
        allowed_classes=allowed_classes
    )

    print(test_dataset.allowed_classes)

    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False, num_workers=4, persistent_workers=True)

    return train_loader, val_loader, test_loader


def get_resnet18_embedder():
    # 1. Load the pretrained ResNet-18 model
    # We use DEFAULT weights which correspond to ImageNet 1K
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # 2. Replace the final fully connected layer with an Identity layer
    # This strips away the classification head, leaving the 512-dimensional embedding
    model.fc = torch.nn.Identity()
    
    # 3. Set the model to evaluation mode (important for BatchNorm layers)
    model.eval()
    
    return model


def count_selected_attributes(images_path, attributes_path, selected_indices):
    # 1. Create the full 312-dim map
    full_map = create_concept_map(images_path, attributes_path)
    
    # 2. Convert 1-based indices to 0-based for slicing
    indices = [i - 1 for i in selected_indices]
    
    total_ones = 0
    num_images = len(full_map)
    
    # 3. Iterate through every image and count 1s in the selected slice
    for filename, full_vector in full_map.items():
        # Slice the vector to only the selected concepts
        selected_vector = full_vector[indices]
        
        # Sum the 1s (since it's a binary tensor, sum equals the count of 1s)
        total_ones += torch.sum(selected_vector).item()

    # 4. Display Results
    avg_per_bird = total_ones / num_images if num_images > 0 else 0

    avgPositives = avg_per_bird / len(indices) if len(indices) > 0 else 0
    
    print("-" * 30)
    print(f"Dataset Summary (Selected Subset)")
    print("-" * 30)
    print(f"Total Images:          {num_images}")
    print(f"Concepts Tracked:      {len(indices)}")
    print(f"Total 1s found:        {int(total_ones)}")
    print(f"Avg 1s per bird:       {avg_per_bird:.2f}")
    print(f"Avg Positives/Bird:    {avgPositives:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    #extract_and_save_embeddings(images_file="/Users/alecvandeuren/Thesis/src/data/raw/CUB/CUB_fixed/images.txt", root_dir='/Users/alecvandeuren/Thesis/src/data/raw/CUB/CUB_BASE/CUB_200_2011/images', output_file="cub_embeddings.pt")
    #train_loader, val_loader, test_loader = createPrecomputedDataloaders(batchSize=32, embeddings_file="cub_embeddings.pt")
    #for images, concepts, labels in train_loader:
    #    print("Batch of images shape:", images.shape)  # Should be [batch_size, 3, 299, 299]
    #    print("Batch of concepts shape:", concepts.shape)  # Should be [batch_size, len(SELECTED_CONCEPTS)]
    #    print("Batch of labels shape:", labels.shape)  # Should be [batch_size, num_classes]
    #    print(concepts[0])
    #    break  # Just check one batch
    count_selected_attributes(
        images_path="/Users/alecvandeuren/Thesis/src/data/raw/CUB/CUB_fixed/images.txt",
        attributes_path="/Users/alecvandeuren/Thesis/src/data/raw/CUB/CUB_fixed/image_attribute_labels.txt",
        selected_indices=SELECTED_CONCEPTS
    )