import numpy as np
import pandas as pd
import os
from config import *

def DISFA_get_subject(sub_ID, preprocess_steps_str, camera = 'left'):
    if camera == 'left':
        DISFA_img_rootdir = DISFA_img_rootdir_leftcam
    elif camera == 'right':
        DISFA_img_rootdir = DISFA_img_rootdir_rightcam
    else:
        raise ValueError('Invalid Camera Direction')

    img_dir = os.path.join(f'{DISFA_img_rootdir}_{preprocess_steps_str}', sub_ID)

    img_names = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    img_names = sorted(img_names, key=lambda x: int(x.split('.')[0]))
    img_paths = [os.path.join(img_dir, name) for name in img_names]
        
    frame_indices = [int(name.split('.')[0]) for name in img_names]
    df_AUs = {}
    for AU in DISFA_AUs:
        label_path = os.path.join(DISFA_label_rootdir, sub_ID, sub_ID + '_au' + str(AU) + '.txt')
        df_AU = pd.read_csv(label_path, header=None, names=['intensity'], index_col=0)
        df_AUs[AU] = df_AU

    labels = []
    for frame_index in frame_indices:
        frame_labels = -np.ones(n_all_AUs) # Use -1 as the placeholder for AUs not labeled in the DISFA dataset 
        for AU in DISFA_AUs:
            if frame_index in df_AUs[AU]['intensity'].index:
                label = df_AUs[AU]['intensity'][frame_index]
                if 0 <= label <= 5:
                    frame_labels[AU-1] = label
        labels.append(frame_labels)
    return img_paths, labels

def DISFAPlus_get_subject(sub_ID, preprocess_steps_str):
    all_img_paths = []
    all_labels = []
    img_rootdir = os.path.join(f'{DISFAPlus_img_rootdir}_{preprocess_steps_str}', sub_ID)
    trials = os.listdir(img_rootdir)
    trials = sorted(trials)
    for trial in trials:
        img_dir = os.path.join(img_rootdir, trial)
        img_names = os.listdir(img_dir)
        img_names = sorted(img_names, key=lambda x: int(x.split('.')[0]))
        img_paths = [os.path.join(img_dir, name) for name in img_names]
        all_img_paths.extend(img_paths)
        
        df_AUs = {}
        for AU in DISFAPlus_AUs:
            label_path = os.path.join(DISFAPlus_label_rootdir, sub_ID, trial, f'AU{AU}.txt')
            df_AU = pd.read_csv(label_path, header=None, delimiter='     ', skiprows=2, names=['intensity'], index_col=0, engine='python')
            df_AUs[AU] = df_AU
        labels = []
        for img_name in img_names:
            frame_labels = -np.ones(n_all_AUs) # Use -1 as the placeholder for AUs not labeled in the DISFA+ dataset 
            for AU in DISFAPlus_AUs:
                label = df_AUs[AU]['intensity'][img_name]
                if 0 <= label <= 5:
                    frame_labels[AU-1] = label
            labels.append(frame_labels)
        all_labels.extend(labels)
    return all_img_paths, all_labels