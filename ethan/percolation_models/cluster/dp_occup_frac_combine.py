"""
Created Oct 10 2024
Updated Oct 10 2024

(IN CLUSTER)
Combine all occupation fractions
"""
import numpy as np
import os

file_names = os.listdir('./')

prob_master = []
occup_frac_master = []
for file_name in file_names:
    if file_name.endswith('.npy') and 'dp_occup' in file_name:
        probs, occup_fracs = np.load(f'./{file_name}')
        prob_master.append(probs)
        occup_frac_master.append(occup_fracs)

prob_master = np.array(prob_master)
occup_frac_master = np.array(occup_frac_master)

occup_frac_master_mean = np.mean(occup_frac_master, axis=0)
prob_master = prob_master[0]

np.save('./dp_occup_frac_mean_size=5000.npy', np.array([prob_master, occup_frac_master_mean]))
