import os
import numpy as np
import pandas as pd
import cv2
import json
from collections import defaultdict
import torch
from scores.CLIPScore import CLIPScore
from scores.BLIPScore import BLIPScore
from scores.AestheticScore import AestheticScore
import ImageReward as RM
from scores.PickScore import PickScore
import hpsv2
from config import *

def main():
    # Assign the scoring models to multiple GPUs
    model_CLIP = CLIPScore(device = "cuda:0").to("cuda:0")
    model_Aesthetic = AestheticScore(device = "cuda:1").to("cuda:1")
    model_BLIP = BLIPScore(device = "cuda:2").to("cuda:2")
    model_ImageReward = RM.load("ImageReward-v1.0", device = "cuda:3").to("cuda:3")
    model_PickScore = PickScore(device = "cuda:1").to("cuda:1")
    
    with open(os.path.join(prep_rootdir, 'valid_sessions.json'), 'r') as f:
        all_valid_sessions = json.load(f)
        
    with open(os.path.join(prep_rootdir, 'invalid_images.json'), 'r') as f:
        all_invalid_images = json.load(f)
        
    # Compute multiple scores for each text-to-image generation and save the results
    for user_id in user_id_list:
        user_rootdir = os.path.join(data_rootdir, user_id)
        valid_sessions = all_valid_sessions[user_id]
        invalid_images = all_invalid_images[user_id]
        
        data_dict = defaultdict(list)

        for session_idx in valid_sessions:
            session_dir = os.path.join(user_rootdir, f"session_{session_idx}")
            with open(os.path.join(session_dir, 'data.json'), 'r') as file:
                data = json.load(file)
            n_images = len([f for f in os.listdir(session_dir) if f.startswith('image') and f.endswith('.png')])
            ranking = list(map(lambda img: img['id'], data['ranking']))
            image_paths = [os.path.join(session_dir, f"image{image_idx}.png") for image_idx in range(1, n_images+1)]
            session_CLIP_scores = model_CLIP.score(data['prompt'], image_paths)
            session_Aesthetic_scores = model_Aesthetic.score(data['prompt'], image_paths)
            session_BLIP_scores = model_BLIP.score(data['prompt'], image_paths)
            session_ImageReward_scores = model_ImageReward.score(data['prompt'], image_paths)
            session_PickScores = model_PickScore.score(data['prompt'], image_paths)
            session_HPS_V2_scores = hpsv2.score(image_paths, data['prompt'])

            for image_idx in range(1, n_images+1):
                if (image_idx) not in invalid_images[str(session_idx)]:
                    data_dict['session_index'].append(session_idx)
                    data_dict['image_index'].append(image_idx)
                    data_dict['overall_rating'].append(data['survey'][image_idx-1]['overallRating'])
                    data_dict['alignment_rating'].append(data['survey'][image_idx-1]['alignmentRating'])
                    data_dict['fidelity_rating'].append(data['survey'][image_idx-1]['fidelityRating'])
                    
                    for issue, value in data['survey'][image_idx-1]['issuesList'].items():
                        if user_id not in participants_with_invalid_issue_responses:
                            data_dict[issue].append(value)
                        else:
                            data_dict[issue].append(np.nan)
                    for reaction, value in data['survey'][image_idx-1]['reactionList'].items():
                        data_dict[reaction].append(value)
                    data_dict['rank_score'].append(n_images - ranking.index(image_idx-1))
                    data_dict['CLIP_score'].append(session_CLIP_scores[image_idx-1])
                    data_dict['Aesthetic_score'].append(session_Aesthetic_scores[image_idx-1])
                    data_dict['BLIP_score'].append(session_BLIP_scores[image_idx-1])
                    data_dict['ImageReward_score'].append(session_ImageReward_scores[image_idx-1])
                    data_dict['PickScore'].append(session_PickScores[image_idx-1])
                    data_dict['HPS_V2_score'].append(session_HPS_V2_scores[image_idx-1])
            print(f"Participant {user_id} Session {session_idx} completed...")

        df_preprocessed_data = pd.DataFrame(data_dict)

        preprocessed_data_folder = os.path.join(prep_rootdir, f"preprocessed_image_data")
        os.makedirs(preprocessed_data_folder, exist_ok=True)
        df_preprocessed_data.to_csv(os.path.join(preprocessed_data_folder, f"{user_id}.csv"), index=False)

if __name__ == '__main__':
    main()