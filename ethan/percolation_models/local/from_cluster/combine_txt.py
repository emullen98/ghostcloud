"""
Created Jun 06 2024
Updated Jun 06 2024

Combine .txt files that contains the exponents for each lattice into a csv
Column names should be:
['jobid', 'taskid', 'area_pdf', 'perim_pdf', 'dvs']
"""

import numpy as np
import pandas as pd
import os

full_dict = {'jobid': [], 'taskid': [],
             'area_pdf': [], 'perim_pdf': [],
             'dvs': []}

file_names = os.listdir('../../data')
for file in file_names:
    if file.endswith('.txt'):
        file_stuff = file.split('_')
        jobid = file_stuff[5][4:]
        taskid = file_stuff[6].split('.')[0][5:]
        f = open(f'../../data/{file}', 'r')
        exps = f.readlines()
        exps = [str(np.round(float(num[:-1]), 3)) for num in exps]
        exps.insert(0, jobid)
        exps.insert(1, taskid)
        for k, v in zip(full_dict, exps):
            full_dict[k].append(v)

df = pd.DataFrame(full_dict)
df.to_csv('../../data/all_exps.csv', header=True)
