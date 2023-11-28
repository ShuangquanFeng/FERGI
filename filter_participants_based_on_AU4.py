import os
import json
import numpy as np
import pandas as pd
from config import *
def main():
    percentile = 80
    threshold = 0.5
    
    excluded_participants = []
    for user_id in user_id_list:
        df = pd.read_csv(os.path.join(prep_rootdir, f'preprocessed_baseline_data', f"{user_id}.csv"))
        AU4_activation_values = df['AU4_activation_value'].to_numpy()
        if np.percentile(AU4_activation_values, percentile) >= threshold:
            excluded_participants.append(user_id)
            
    with open(os.path.join(prep_rootdir, 'AU4_excluded_participants.json'), 'w') as f:
        json.dump(excluded_participants, f, indent=4)

if __name__ == '__main__':
    main()