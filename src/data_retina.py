import os
import pandas as pd
from PIL import Image
from pathlib import Path
import re

class RetinaDataset:
    def __init__(self, base_dir, dataset_name, split="all", filter_macula=True):
        self.base_dir = Path(base_dir)
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.df = None
        self.images_dir = None
        self.filter_macula = filter_macula
        
        self.load_dataset()
        
    def load_dataset(self):
        if self.dataset_name == "brset":
            path = self.base_dir / 'BRSET' / 'brset'
            self.df = pd.read_csv(path / 'labels_brset.csv')
            
            # Load splits
            split_path = path / 'labels_splits.csv'
            if split_path.exists():
                split_df = pd.read_csv(split_path)
                # Merge on patient_id and image_id to get 'split' column
                self.df = self.df.merge(split_df, on=['patient_id', 'image_id'])
            
            self.images_dir = path / 'images_224'
            # BRSET image filenames usually match image_id + .jpg
            # Ensuring consistency
            if 'image_id' in self.df.columns:
                self.df['image_path'] = self.df['image_id'].apply(lambda x: str(self.images_dir / f"{x}.jpg"))

            self.df['Task_Referable'] = ((self.df['DR_ICDR'] >= 2) | (self.df['macular_edema'] == 1)).astype(int)
        
            # BRSET Tasks
            # 5 Class: DR_ICDR (0-4)
            
            # 3 Class: 0 -> 0, 1-3 -> 1, 4 -> 2
            def map_3_class(val):
                try:
                    v = int(val)
                    if v == 0: return 0
                    if 1 <= v <= 3: return 1
                    if v == 4: return 2
                except: pass
                return -1
        
            if 'DR_ICDR' in self.df.columns:
                self.df['Task_3_Classes'] = self.df['DR_ICDR'].apply(map_3_class)
                self.df['DR_2_Class'] = self.df['DR_ICDR'].apply(lambda x: 0 if x == 0 else (1 if 1 <= x <= 4 else -1))
        
            # Glaucoma Proxy: increased_cup_disc
            if 'increased_cup_disc' in self.df.columns:
                self.df['Task_Glaucoma'] = self.df['increased_cup_disc']
        
            # AMD
            if 'amd' in self.df.columns:
                self.df['Task_AMD'] = self.df['amd']
            
             
        elif self.dataset_name == "mbrset":
            path = self.base_dir / 'mBRSET' / 'mbrset'
            self.df = pd.read_csv(path / 'labels_mbrset.csv')
            
            # Load splits
            split_path = path / 'labels_splits.csv'
            if split_path.exists():
                split_df = pd.read_csv(split_path)
                # Merge on patient and file
                self.df = self.df.merge(split_df, on=['patient', 'file'])

            self.images_dir = path / 'images_224'
            
            
            IMG_COL = "file"
            DR_COL = "final_icdr"
            EDEMA_COL = "final_edema"
            
            # Filter Macula Images: Keep only those with .1.jpg or .3.jpg (frontal macula)
            if self.filter_macula:
                macula_regex = re.compile(r'\.[13](\.jpg)?$', re.IGNORECASE)
                self.df = self.df[self.df[IMG_COL].astype(str).apply(lambda x: bool(macula_regex.search(x)))].copy()
            
            # Standardizing Labels
            if EDEMA_COL in self.df.columns:
                self.df[EDEMA_COL] = self.df[EDEMA_COL].astype(str).str.lower().str.strip()
                self.df['edema_bin'] = self.df[EDEMA_COL].map({'yes': 1, 'no': 0}).fillna(0).astype(int)

            self.df[DR_COL] = pd.to_numeric(self.df[DR_COL], errors='coerce')
            self.df = self.df.dropna(subset=[DR_COL])
            
            # Task columns
            self.df['Task_Referable'] = ((self.df[DR_COL] >= 2) | (self.df['edema_bin'] == 1)).astype(int)

            # binary DR: 0 -> 0, 1-4 -> 1
            self.df['Task_3_Classes'] = self.df[DR_COL].apply(lambda x: 0 if x == 0 else (1 if 1 <= x <= 4 else -1))
            self.df['DR_2_Class'] = self.df[DR_COL].apply(lambda x: 0 if x == 0 else (1 if 1 <= x <= 4 else -1))
            
            # Image path
            self.df['image_path'] = self.df[IMG_COL].apply(lambda x: str(self.images_dir / x))

        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        # Filter by split if applicable
        if self.split != "all":
            if "split" in self.df.columns:
                print(f"Filtering dataset for split: {self.split}")
                self.df = self.df[self.df["split"] == self.split].copy()
                if len(self.df) == 0:
                    print(f"Warning: No samples found for split '{self.split}'")
            else:
                print(f"Warning: Split '{self.split}' requested but 'split' column not found in dataset.")

    def __len__(self):
        return len(self.df)

    def get_row(self, idx):
        return self.df.iloc[idx]

    def get_image(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        try:
            return Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None
