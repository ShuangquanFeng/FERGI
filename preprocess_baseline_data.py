import os
import numpy as np
import pandas as pd
import cv2
import json
from collections import defaultdict
from config import *
from utils import *

def main():
    with open(os.path.join(prep_rootdir, 'valid_sessions.json'), 'r') as f:
        all_valid_sessions = json.load(f)
        
    # Compute the AU activation values for baseline clips
    for user_id in user_id_list:
        user_rootdir = os.path.join(data_rootdir, user_id)
        valid_sessions = all_valid_sessions[user_id]
        
        data_dict = defaultdict(list)

        for session_idx in valid_sessions:
            session_dir = os.path.join(user_rootdir, f"session_{session_idx}")
            n_images = len([f for f in os.listdir(session_dir) if f.startswith('image') and f.endswith('.png')])

            for image_idx in range(1, n_images+1):
                df_baseline = pd.read_csv(os.path.join(session_dir, f"baseline_clip_{image_idx}_features.csv"))
                AUs = [col for col in df_baseline.columns if col.startswith('AU')]
                if np.mean(select_valid_frames(df_baseline)) >= valid_proportion_threshold:
                    data_dict['session_index'].append(session_idx)
                    data_dict['image_index'].append(image_idx)
                    
                    df_baseline['frame_is_valid'] = select_valid_frames(df_baseline)
                    df_baseline_filtered = df_baseline[df_baseline['frame_is_valid']].copy()
                    
                    moving_window_size = int(((1 / np.mean(np.diff(df_baseline['timestamp']))).round() / 10).round())
                    df_baseline_moving_window_mean = df_baseline.copy()
                    for AU in AUs:
                        df_baseline_moving_window_mean[AU] = df_baseline_moving_window_mean[AU].rolling(window=moving_window_size).mean()
                    df_baseline_moving_window_mean['frame_is_valid'] = df_baseline_moving_window_mean['frame_is_valid'].rolling(window=moving_window_size).min().fillna(False).astype(bool)
                    df_baseline_moving_window_mean_filtered = df_baseline_moving_window_mean[df_baseline_moving_window_mean['frame_is_valid']].copy()
                    for AU in AUs:
                        data_dict[f"{AU}_activation_value"].append(df_baseline_moving_window_mean_filtered[AU].max() - df_baseline_moving_window_mean_filtered[AU].iloc[0])

        df_preprocessed_data = pd.DataFrame(data_dict)

        preprocessed_data_folder = os.path.join(prep_rootdir, f'preprocessed_baseline_data')
        os.makedirs(preprocessed_data_folder, exist_ok=True)
        df_preprocessed_data.to_csv(os.path.join(preprocessed_data_folder, f"{user_id}.csv"), index=False)
       

if __name__ == '__main__':
    main()