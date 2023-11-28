import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import warnings
from AU_datasets_preparation import *

class AU_dataset(Dataset):
    def __init__(self, subject_dict, preprocess_steps_str, transform=None):
        all_img_paths = []
        all_labels = []

        # Access images from DISFA+ dataset
        if 'DISFAPlus' in subject_dict:
            sub_IDs = subject_dict['DISFAPlus']
            for sub_ID in sub_IDs:
                img_paths, labels = DISFAPlus_get_subject(sub_ID, preprocess_steps_str)
                all_img_paths.extend(img_paths)
                all_labels.extend(labels)
        
        # Access images from the left camera in DISFA dataset
        if 'DISFAleft' in subject_dict:
            sub_IDs = subject_dict['DISFAleft']
            for sub_ID in sub_IDs:
                img_paths, labels = DISFA_get_subject(sub_ID, preprocess_steps_str, camera = 'left')
                all_img_paths.extend(img_paths)
                all_labels.extend(labels)
        
        # Access images from the right camera in DISFA dataset
        if 'DISFAright' in subject_dict:
            sub_IDs = subject_dict['DISFAright']
            for sub_ID in sub_IDs:
                img_paths, labels = DISFA_get_subject(sub_ID, preprocess_steps_str, camera = 'right')
                all_img_paths.extend(img_paths)
                all_labels.extend(labels)
        
        self.all_img_paths = all_img_paths
        self.all_labels = all_labels
        self.transform = transform

    def __len__(self):
        return len(self.all_img_paths)

    def __getitem__(self, index):
        path = self.all_img_paths[index]
        img = cv2.imread(path)
        label = self.all_labels[index]
        if self.transform:
            img = self.transform(img)
        warnings.filterwarnings("ignore", category=UserWarning)
        return path, torch.tensor(img, dtype=torch.float), torch.tensor(label, dtype=torch.float)