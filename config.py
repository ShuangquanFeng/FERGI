import os

n_all_AUs = 64
max_intensity = 5

# Names of all AUs
action_units = {
    1: "Inner Brow Raiser",
    2: "Outer Brow Raiser",
    4: "Brow Lowerer",
    5: "Upper Lid Raiser",
    6: "Cheek Raiser",
    7: "Lid Tightener",
    9: "Nose Wrinkler",
    10: "Upper Lip Raiser",
    11: "Nasolabial Deepener",
    12: "Lip Corner Puller",
    13: "Cheek Puffer",
    14: "Dimpler",
    15: "Lip Corner Depressor",
    16: "Lower Lip Depressor",
    17: "Chin Raiser",
    18: "Lip Puckerer",
    20: "Lip stretcher",
    22: "Lip Funneler",
    23: "Lip Tightener",
    24: "Lip Pressor",
    25: "Lips part",
    26: "Jaw Drop",
    27: "Mouth Stretch",
    28: "Lip Suck",
    41: "Lid droop",
    42: "Slit",
    43: "Eyes Closed",
    44: "Squint",
    45: "Blink",
    46: "Wink",
    51: "Head Turn Left",
    52: "Head Turn Right",
    53: "Head Up",
    54: "Head Down",
    55: "Head Tilt Left",
    56: "Head Tilt Right",
    57: "Head Forward",
    58: "Head Back",
    61: "Eyes Turn Left",
    62: "Eyes Turn Right",
    63: "Eyes Up",
    64: "Eyes Down"
}

projects_rootdir = '..'

# The directories of the DISFA dataset
DISFA_rootdir = os.path.join(projects_rootdir, 'FER_datasets/DISFA')
DISFA_img_rootdir_leftcam = os.path.join(DISFA_rootdir, 'Images_LeftCamera')
DISFA_img_rootdir_rightcam = os.path.join(DISFA_rootdir, 'Images_RightCamera')
DISFA_label_rootdir = os.path.join(DISFA_rootdir, 'ActionUnit_Labels')
DISFA_subjects = os.listdir(DISFA_label_rootdir)
DISFA_AUs = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]

# The directories of the DISFA+ dataset
DISFAPlus_rootdir = os.path.join(projects_rootdir, 'FER_datasets/DISFAPlus')
DISFAPlus_img_rootdir = os.path.join(DISFAPlus_rootdir, 'Images')
DISFAPlus_label_rootdir = os.path.join(DISFAPlus_rootdir, 'Labels')
DISFAPlus_subjects = [d for d in os.listdir(DISFAPlus_label_rootdir) if os.path.isdir(os.path.join(DISFAPlus_label_rootdir, d))]
DISFAPlus_AUs = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]

data_rootdir = 'data'
prep_rootdir = 'preparation'
results_rootdir = 'results'
figures_rootdir = 'figures'

valid_proportion_threshold = 0.8 # Threshold that the proportion of valid frames in a clip needs to meet to make the clip valid
FDCS_min = 0.9 # Facial detection confidence score
PIR_min, PIR_max = 0.55, 0.85 # Pitch indicative score
YIR_min, YIR_max = 0.3, 0.7 # Yaw indicative score

issue_options = ['unmentioned_content', 'repeated_generation', 'existence_of_body', 'blurry', 'psychological_discomfort', 'sexual_content', 'violent_content', 'defamatory_content']
reaction_options = ['disappointed', 'satisfied', 'surprised', 'disgusted', 'amused', 'scared']

user_id_list = ['001', '002', '004', '005', '006', '007', '008', '009', '010', '011', '012', '015', '016', '017', '018', '020', '021', '022', '023', '024', '025', '026', '028', '030', '031', '032', '033', '034', '035', '037', '038', '039', '040'] # The user IDs of all 33 participants included in the dataset
participants_with_invalid_issue_responses = ['001', '002', '004', '005', '006', '007', '008'] # The question regarding issues of the image had a bug for these participants so that the data of their responses to this question should not be used