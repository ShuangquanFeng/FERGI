import numpy as np
import cv2
import os
import json
import csv
import argparse
import torch
from config import *
from utils import *
from preprocess_datasets import *
from iresnet_modified import *

parser = argparse.ArgumentParser(description='Process the AU intensities of baseline clips and reaction clips')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')

def main():
    args = parser.parse_args()
    gpu = args.gpu

    # Load the model and its setup
    model_path = 'AU_models/checkpoint_epoch3.pth'
    setup_path = 'AU_models/training_setup.pkl'
    
    with open(setup_path, 'rb') as f:
        setup = pkl.load(f)
    model_name = setup['model']['name']
    
    preprocess_steps = setup['preprocess']
    transform_steps = setup['transform']
    data_transform = transforms_compose(transform_steps, include_random=False)
    
    sel_AUs = np.array(setup['AUs'])
    
    with open(os.path.join(prep_rootdir, 'valid_sessions.json'), 'r') as f:
        all_valid_sessions = json.load(f)
    
    print("Initializing the model...")
    device = torch.device(f"cuda:{gpu}")
    if model_name == 'IR50':
        model = iresnet_basic(n_layers=50, n_features = n_all_AUs * (max_intensity + 1)).to(device)
        for param in model.parameters():
            param.requires_grad = False
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        ValueError('Unknown model option.')
    
    # Process the facial features (estimate and save the AU intensity of each frame in the baseline clips and reaction clips)
    model.eval()
    with torch.no_grad():
        for user_id in user_id_list:
            user_rootdir = os.path.join(data_rootdir, user_id)
            valid_sessions = all_valid_sessions[user_id]
            for session_idx in valid_sessions:
                session_dir = os.path.join(user_rootdir, f"session_{session_idx}")
                n_clips = len([f for f in os.listdir(session_dir) if f.startswith('reaction_clip_') and f.endswith('.mp4')])

                # Process baseline clips
                for clip_idx in range(1, n_clips+1):
                    clip_load_path = os.path.join(session_dir, f"baseline_clip_{clip_idx}.mp4")
                    capture = cv2.VideoCapture(clip_load_path)

                    all_frames = []
                    all_secs = []
                    all_face_detection_confidence_scores = []
                    all_pitch_indicative_ratios = []
                    all_yaw_indicative_ratios = []
                    all_AU_scores = []
                    frame_count = 0
                    while True:
                        check, frame = capture.read()
                        if check == False:
                            break
                        frame_count += 1
                        all_frames.append(frame_count)
                        all_secs.append(capture.get(cv2.CAP_PROP_POS_MSEC) / 1000)
                        frame, face_detection_confidence_score, pitch_indicative_ratio, yaw_indicative_ratio = preprocess_image(frame, preprocess_steps, device)
                        all_face_detection_confidence_scores.append(face_detection_confidence_score)
                        all_pitch_indicative_ratios.append(pitch_indicative_ratio)
                        all_yaw_indicative_ratios.append(yaw_indicative_ratio)
                        if face_detection_confidence_score > 0:
                            inputs = data_transform(frame).unsqueeze(0).to(device)
                            outputs = model(inputs)
                            AU_intensities = outputs[0,:n_all_AUs].cpu().numpy()
                            all_AU_scores.append(AU_intensities)
                        else:
                            all_AU_scores.append(None)

                    clip_save_path = os.path.join(session_dir, f"baseline_clip_{clip_idx}_features.csv")
                    with open(clip_save_path, 'w', newline = '') as file:
                        writer = csv.writer(file)
                        writer.writerow(["frame", "timestamp", "face_detection_confidence_score", "pitch_indicative_ratio", "yaw_indicative_ratio"] + [f"AU{AU}" for AU in sel_AUs])
                        for i in range(len(all_frames)):
                            if all_face_detection_confidence_scores[i] > 0:
                                writer.writerow([all_frames[i], f"{all_secs[i]:.3f}", f"{all_face_detection_confidence_scores[i]:.2f}", f"{all_pitch_indicative_ratios[i]:.2f}", f"{all_yaw_indicative_ratios[i]:.2f}"] + list(map(lambda x: f"{x:.2f}", all_AU_scores[i][np.array(sel_AUs)-1])))
                            else:
                                writer.writerow([all_frames[i], f"{all_secs[i]:.3f}", "0", "-1", "-1"] + [-1] * len(sel_AUs))

                # Process reaction clips
                for clip_idx in range(1, n_clips+1):
                    clip_load_path = os.path.join(session_dir, f"reaction_clip_{clip_idx}.mp4")
                    capture = cv2.VideoCapture(clip_load_path)

                    all_frames = []
                    all_secs = []
                    all_face_detection_confidence_scores = []
                    all_pitch_indicative_ratios = []
                    all_yaw_indicative_ratios = []
                    all_AU_scores = []
                    frame_count = 0
                    while True:
                        check, frame = capture.read()
                        if check == False:
                            break
                        frame_count += 1
                        all_frames.append(frame_count)
                        all_secs.append(capture.get(cv2.CAP_PROP_POS_MSEC) / 1000)
                        frame, face_detection_confidence_score, pitch_indicative_ratio, yaw_indicative_ratio = preprocess_image(frame, preprocess_steps, device)
                        all_face_detection_confidence_scores.append(face_detection_confidence_score)
                        all_pitch_indicative_ratios.append(pitch_indicative_ratio)
                        all_yaw_indicative_ratios.append(yaw_indicative_ratio)
                        if face_detection_confidence_score > 0:
                            inputs = data_transform(frame).unsqueeze(0).to(device)
                            outputs = model(inputs)
                            AU_intensities = outputs[0,:n_all_AUs].cpu().numpy()
                            all_AU_scores.append(AU_intensities)
                        else:
                            all_AU_scores.append(None)

                    clip_save_path = os.path.join(session_dir, f"reaction_clip_{clip_idx}_features.csv")
                    with open(clip_save_path, 'w', newline = '') as file:
                        writer = csv.writer(file)
                        writer.writerow(["frame", "timestamp", "face_detection_confidence_score", "pitch_indicative_ratio", "yaw_indicative_ratio"] + [f"AU{AU}" for AU in sel_AUs])
                        for i in range(len(all_frames)):
                            if all_face_detection_confidence_scores[i] > 0:
                                writer.writerow([all_frames[i], f"{all_secs[i]:.3f}", f"{all_face_detection_confidence_scores[i]:.3f}", f"{all_pitch_indicative_ratios[i]:.3f}", f"{all_yaw_indicative_ratios[i]:.3f}"] + list(map(lambda x: f"{x:.3f}", all_AU_scores[i][np.array(sel_AUs)-1])))
                            else:
                                writer.writerow([all_frames[i], f"{all_secs[i]:.3f}", "0", "-1", "-1"] + [-1] * len(sel_AUs))
                print(f"Participant {user_id} Session {session_idx} completed...")
if __name__ == '__main__':
    main()