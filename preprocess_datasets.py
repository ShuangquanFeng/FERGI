import os
import numpy as np
import cv2

# For encode and decode the preprocessing steps
import pickle as pkl
import base64
import zlib

# Mediapipe for face cropping
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='pretrained_models/blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# For face alignment
import PIL
import PIL.Image
import scipy

# MobileFaceNet for face alignment 
import torch
from torchvision import transforms
from mobilefacenet import MobileFaceNet
np2torch_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((112, 112)), transforms.ToTensor(),])
landmark_model = MobileFaceNet()
weights_path = 'pretrained_models/mobilefacenet_model_best.pth.tar'
weights = torch.load(weights_path)['state_dict']
landmark_model.load_state_dict(weights)
landmark_model.eval()

def preprocess_datasets(src_folder, steps, device_as_parameter):
    # Encode the preprocessing steps
    steps_str = encode_steps(steps)
    dest_folder = f"{src_folder}_{steps_str}"

    # Return if the dataset has already been preprocessed with the specified steps
    if os.path.exists(dest_folder):
        return steps_str
    
    # Preprocess each image and save the preprocessed version
    os.makedirs(dest_folder, exist_ok=True)
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                src_path = os.path.join(root, file)

                relative_path = os.path.relpath(root, src_folder)
                dest_dir = os.path.join(dest_folder, relative_path)
                os.makedirs(dest_dir, exist_ok=True)
                
                dest_path = os.path.join(dest_dir, file)
                
                img = cv2.imread(src_path)
                img, face_detection_confidence_score, pitch_indicative_ratio, yaw_indicative_ratio = preprocess_image(img, steps, device_as_parameter)

                if face_detection_confidence_score > 0:
                    cv2.imwrite(dest_path, img)
    return steps_str
                    
def encode_steps(steps):
    return base64.urlsafe_b64encode(zlib.compress(pkl.dumps(steps))).decode('utf-8')

def decode_steps(steps_str):
    return pkl.loads(zlib.decompress(base64.urlsafe_b64decode(steps_str.encode('utf-8'))))
    
def preprocess_image(img, steps, device_as_parameter):
    global device
    device = device_as_parameter
    global landmark_model
    landmark_model = landmark_model.to(device)
    
    face_detection_confidence_score = 0
    pitch_indicative_ratio = -1
    yaw_indicative_ratio = -1
    for process, inp in steps:
        if img is None or img.size == 0:
            return False, None
        if process == 'cvtColor': # Conversion between BGR and RGB
            img = convert_color(img, inp)
        elif process == 'cropping': # Face cropping
            img, face_detection_confidence_score = crop_image(img, inp[0], inp[1])
            if face_detection_confidence_score == 0:
                return img, face_detection_confidence_score, pitch_indicative_ratio, yaw_indicative_ratio
        elif process == 'alignment': # Face alignment
            img, pitch_indicative_ratio, yaw_indicative_ratio = align_image(img, inp)
        elif process == 'heNlm': # Combination of histogram equalization and linear mapping
            img = weighted_histogram_equalization_and_linear_mapping(img, inp)
        elif process == 'padding': # Image padding
            img = pad_image(img)
        elif process == 'resize': # Image resizing
            img = resize_image(img, inp)
        else:
            raise ValueError('Unrecognized process')
    return img, face_detection_confidence_score, pitch_indicative_ratio, yaw_indicative_ratio

def convert_color(img, cvt): # Conversion between BGR and RGB
    if cvt == 'BGR2RGB':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif cvt == 'RGB2BGR':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError('Unrecognized color conversion')
    return img

def align_image(img, output_size): # Face alignment
    img = align_face(img, output_size=output_size)
    return img
        
def crop_image(img, backend, margins): # Face cropping
    if backend == 'mediapipe':
        detection_result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img))
        if len(detection_result.detections) == 0:
            return img, 0
        
        main_face_idx = np.argmax([detection_result.detections[i].bounding_box.width for i in range(len(detection_result.detections))])
        detection = detection_result.detections[main_face_idx]
        score = detection_result.detections[main_face_idx].categories[0].score
        
        width = detection.bounding_box.width
        height = detection.bounding_box.height
        xmin = detection.bounding_box.origin_x
        xmax = xmin + width
        ymin = detection.bounding_box.origin_y
        ymax = ymin + height
    else:
        raise ValueError('Unrecognized face cropping backend')
        
    top_margin = int(margins['ymin']*height)
    bottom_margin = int(margins['ymax']*height)
    left_margin = int(margins['xmin']*width)
    right_margin = int(margins['xmax']*width)
    
    ymin = ymin + top_margin
    ymax = ymax + bottom_margin
    xmin = xmin + left_margin
    xmax = xmax + right_margin
    
    if ymin < 0 or ymax > img.shape[0] or xmin < 0 or xmax > img.shape[1]:
        img = cv2.copyMakeBorder(img, height, height, width, width, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        ymin += height
        ymax += height
        xmin += width
        xmax += width

    img = img[ymin:ymax, xmin:xmax]
    return img, score

def weighted_histogram_equalization_and_linear_mapping(img, weights): # Combination of histogram equalization and linear mapping
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    
    img_he = img.copy()
    img_he[:,:,0] = cv2.equalizeHist(img_he[:,:,0])
    
    img_lm = img.copy().astype(np.float64)
    img_lm[:,:,0] = (img_lm[:,:,0] - img_lm[:,:,0].min()) / (img_lm[:,:,0].max() - img_lm[:,:,0].min()) * 255
    
    img = ((weights[0] * img_he.astype(np.float64) + weights[1] * img_lm.astype('float')) / np.sum(weights)).astype(np.uint8)
    
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    return img

def pad_image(img): # Image padding
    """
    Pads img (a cv2 image) to make it square with black background.
    Returns a new cv2 image.
    """
    height, width = img.shape[:2]
    size = max(height, width)
    
    top_pad = (size - height) // 2
    bottom_pad = size - height - top_pad
    left_pad = (size - width) // 2
    right_pad = size - width - left_pad

    return cv2.copyMakeBorder(img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])

def resize_image(img, new_size): # Image resizing
    img = cv2.resize(img, new_size)
    return img


def get_landmark(img, output_size = 112): # Get facial landmarks
    img = np2torch_transform(img).unsqueeze(0)
    img = img.to(device)
    with torch.no_grad():
        lm = landmark_model(img)
    lm = lm[0].reshape(68, 2).cpu().numpy()
    return lm

def align_face(img, output_size=112): # Face alignment adapted from https://github.com/NVlabs/ffhq-dataset
    lm = get_landmark(img)
    lm = lm * np.array([img.shape[1], img.shape[0]])
    lm_chin = lm[0: 17]  # left-right
    lm_eyebrow_left = lm[17: 22]  # left-right
    lm_eyebrow_right = lm[22: 27]  # left-right
    lm_nose = lm[27: 31]  # top-down
    lm_nostrils = lm[31: 36]  # top-down
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
    lm_mouth_inner = lm[60: 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    nostrils_avg = np.mean(lm_nostrils, axis=0)
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    y = np.flipud(x) * [-1, 1]
    A = np.vstack((x, y)).T
    xscale_left, _ = np.linalg.solve(A, eye_avg - lm[0])
    xscale_right, _ = np.linalg.solve(A, lm[16] - eye_avg)
    _, yscale_down = np.linalg.solve(A, lm[9] - eye_avg)
    
    eye_distance, _ = np.linalg.solve(A, eye_right - eye_left)
    _, nostrils_to_eyes_center = np.linalg.solve(A, nostrils_avg - eye_avg)
    left_eye_to_left_edge, _ = np.linalg.solve(A, eye_left - lm[0])
    right_eye_to_right_edge, _ = np.linalg.solve(A, lm[16] - eye_right)
    pitch_indicative_ratio = nostrils_to_eyes_center / eye_distance
    yaw_indicative_ratio = left_eye_to_left_edge / (left_eye_to_left_edge + right_eye_to_right_edge)
    
    yscale_up = yscale_down/2
    yscale_down = yscale_down*1.05
    quad = np.stack([eye_avg - xscale_left*x - yscale_up*y,
                     eye_avg - xscale_left*x + yscale_down*y,
                     eye_avg + xscale_right*x + yscale_down*y,
                     eye_avg + xscale_right*x - yscale_up*y])
    ysize, xsize = np.linalg.norm(np.diff(quad[:3,:], axis=0), axis=1)
    if xsize >= ysize:
        output_xsize = output_size
        output_ysize = output_size * (ysize/xsize)
    else:
        output_ysize = output_size
        output_xsize = output_size * (xsize/ysize)
    qsize = max(xsize, ysize)
    
    # Read image
    img = PIL.Image.fromarray(img)

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = 0
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Transform.
    img = img.transform((int(output_xsize), int(output_ysize)), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    img = np.array(img)

    return img, pitch_indicative_ratio, yaw_indicative_ratio