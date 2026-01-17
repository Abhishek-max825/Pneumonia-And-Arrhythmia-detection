import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import os

# --- X-Ray Preprocessing ---

class ApplyCLAHE:
    """Custom Transform to apply CLAHE to PIL Images"""
    def __init__(self, clip_limit=5.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img_np = np.array(img)
        # Handle Grayscale vs RGB
        if len(img_np.shape) == 2:
            gray = img_np
            is_rgb = False
        else:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            is_rgb = True
            
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        enhanced = clahe.apply(gray)
        
        if is_rgb:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
            
        return Image.fromarray(enhanced)

def get_xray_transforms(is_train=True):
    # ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # Common transforms
    resize = transforms.Resize((224, 224))
    clahe = ApplyCLAHE(clip_limit=5.0) # Match app.py settings
    to_tensor = transforms.ToTensor()
    
    if is_train:
        return transforms.Compose([
            resize,
            clahe, # Apply CLAHE before augmentation
            transforms.RandomRotation(15), # Reduced rotation
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)), # Milder affine
            transforms.RandomHorizontalFlip(),
            # Removed ColorJitter as CLAHE handles contrast
            to_tensor,
            normalize
        ])
    else:
        return transforms.Compose([
            resize,
            clahe, # Apply CLAHE to test data too!
            to_tensor,
            normalize
        ])

def get_xray_loaders(data_dir, batch_size=32, num_workers=2):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    train_dataset = datasets.ImageFolder(train_dir, transform=get_xray_transforms(is_train=True))
    # We might want to use 'test' as validation if 'val' is too small (Kaggle dataset 'val' is often tiny like 16 images)
    # Let's check sizes later. For now, standard loading.
    if os.path.exists(val_dir) and len(os.listdir(val_dir)) > 0:
         val_dataset = datasets.ImageFolder(val_dir, transform=get_xray_transforms(is_train=False))
    else:
         val_dataset = datasets.ImageFolder(test_dir, transform=get_xray_transforms(is_train=False)) # Fallback
         
    test_dataset = datasets.ImageFolder(test_dir, transform=get_xray_transforms(is_train=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_dataset.class_to_idx

# --- ECG Preprocessing (PTB-XL) ---
import wfdb
import ast

class PTBXLDataset(Dataset):
    def __init__(self, data_dir, transform=None, mode='train'):
        """
        Args:
            data_dir (string): Root directory contenting ptbxl_database.csv and records100/
            mode (string): 'train', 'val', or 'test'
        """
        self.data_dir = data_dir
        self.db_path = os.path.join(data_dir, 'ptbxl_database.csv')
        self.transform = transform
        
        # Load and preprocess metadata
        self.df = pd.read_csv(self.db_path, index_col='ecg_id')
        self.df.scp_codes = self.df.scp_codes.apply(lambda x: ast.literal_eval(x))
        
        # Hardcoded aggregation mapping (Standard PTB-XL Superclasses)
        # Based on physionet.org/content/ptb-xl/1.0.3/scp_statements.csv
        self.agg_df = {
            'NORM': 'NORM', 
            'MI': 'MI', 'AMI': 'MI', 'IMI': 'MI', 'LMI': 'MI', 'ILMI': 'MI', 'IPMI': 'MI', 'ASMI': 'MI', 'ALMI': 'MI', 'INJAS': 'MI', 'INJAL': 'MI', 'INJLA': 'MI', 'INJIN': 'MI',
            'STTC': 'STTC', 'NST_': 'STTC', 'ISC_': 'STTC', 'ISCIN': 'STTC', 'ISCAL': 'STTC', 'ISCAN': 'STTC', 'ISCAS': 'STTC', 'ISCIL': 'STTC', 'ISCLA': 'STTC', 'STD_': 'STTC', 'NDT': 'STTC', 'LNGQT': 'STTC',
            'CD': 'CD', 'LAFB': 'CD', 'IRBBB': 'CD', '1AVB': 'CD', 'IVCD': 'CD', 'CRBBB': 'CD', 'CLBBB': 'CD', 'LPFB': 'CD', 'WPW': 'CD', 'ILBBB': 'CD', '3AVB': 'CD', '2AVB': 'CD',
            'HYP': 'HYP', 'LVH': 'HYP', 'LAO/LAE': 'HYP', 'RVH': 'HYP', 'RAO/RAE': 'HYP', 'SEHYP': 'HYP'
        }
        
        # Add diagnostic_superclass column
        self.df['diagnostic_superclass'] = self.df.scp_codes.apply(self.aggregate_diagnostic)
        
        # Train-Test Split (strat_fold column: 1-8 train, 9 val, 10 test)
        if mode == 'train':
            self.df = self.df[self.df.strat_fold < 9]
        elif mode == 'val':
            self.df = self.df[self.df.strat_fold == 9]
        elif mode == 'test':
            self.df = self.df[self.df.strat_fold == 10]
            
        # Reset index for __getitem__ access
        self.df = self.df.reset_index()
        
        # FILTER MISSING FILES
        # The dataset might be partial. We must verify existence of files to avoid Crashing/Spamming.
        print(f"Dataset ({mode}): Filtering missing files from {len(self.df)} potential records...")
        valid_indices = []
        for idx, row in self.df.iterrows():
            filename = row.filename_lr
            # wfdb needs .hea and .dat. Check existence of header at least.
            header_path = os.path.join(self.data_dir, filename + '.hea')
            dat_path = os.path.join(self.data_dir, filename + '.dat')
            if os.path.exists(header_path) and os.path.exists(dat_path):
                valid_indices.append(idx)
        
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        print(f"Dataset ({mode}): {len(self.df)} records verified and kept.")

    def aggregate_diagnostic(self, y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in self.agg_df:
                tmp.append(self.agg_df[key])
        return list(set(tmp))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load Signal
        # filename_lr example: records100/00000/00001_lr
        filename = self.df.iloc[idx].filename_lr
        full_path = os.path.join(self.data_dir, filename)
        
        # Read with WFDB (returns signal, metadata)
        try:
            signal, _ = wfdb.rdsamp(full_path) # Shape: (1000, 12)
        except Exception as e:
            print(f"Error reading {full_path}: {e}")
            # Return dummy zero signal if read fails
            signal = np.zeros((1000, 12), dtype=np.float32)

        # 2. Process Signal
        # Transpose to (Channels, Length) -> (12, 1000) for PyTorch Conv1d
        signal = signal.transpose()
        signal = torch.tensor(signal, dtype=torch.float32)
        
        # 3. Process Label (Binary for now: NORM vs Abnormal)
        # Or Multi-Label? Let's stick to user request: "Disease class"
        # Let's do Binary Classification first: Normal (NORM) vs Pathology (Anything else)
        
        labels_list = self.df.iloc[idx].diagnostic_superclass
        is_normal = 'NORM' in labels_list
        label = torch.tensor(0.0 if is_normal else 1.0, dtype=torch.float32) # 0=Normal, 1=Abnormal
        
        if self.transform:
            signal = self.transform(signal)
            
        return signal, label

class ECGAugmentation:
    """Randomly apply augmentations to ECG signal"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        # x is Tensor (12, 1000)
        if torch.rand(1) > self.p:
            return x
            
        # 1. Random Scaling (Amplitude shift)
        scale = 1.0 + (torch.rand(1) - 0.5) * 0.2 # 0.9 to 1.1
        x = x * scale
        
        # 2. Random Noise
        noise = torch.randn_like(x) * 0.05
        x = x + noise
        
        # 3. Random Time Shift (Cyclic Roll)
        shift = int((torch.rand(1) - 0.5) * 100) # +/- 50 samples
        x = torch.roll(x, shift, dims=1)
        
        return x

def get_ecg_loaders(data_dir, batch_size=32, num_workers=0):
    # Transforms
    train_transform = ECGAugmentation(p=0.5)
    
    train_dataset = PTBXLDataset(data_dir, transform=train_transform, mode='train')
    # Validation/Test should NOT have augmentation
    val_dataset = PTBXLDataset(data_dir, mode='val')
    test_dataset = PTBXLDataset(data_dir, mode='test')
    
    # Windows: num_workers=0 is safer to avoid multiprocessing issues
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
