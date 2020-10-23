"""
Exp 0: Prepare the data for experiments
"""

import sys
sys.path.append('../../src/')
from ml.dataset import QuakeDataSet
from utils import save_file_jl, load_file_jl

'''
Give relevant file info in the form of list of dictionaries 
The different key values represent function parameters in dataset.py functions 
'''
ld_files = [{'file_name': '../../data/classification_data_u15.txt',
             'training_folder': 'User15',
             'folder_type': 'trimmed_data', 'avg': False, 'load_img': True},

            {'file_name': '../../data/classification_data_u100.txt',
             'training_folder': 'User100',
            'folder_type': 'trimmed_data', 'avg': False, 'load_img': True
             },

            {'file_name': '../../data/classification_data_u0.txt',
             'training_folder': 'User0',
             'folder_type': 'trimmed_data', 'avg': False, 'load_img': True
             },

            {'file_name': '../../data/classification_data_u19.txt',
             'training_folder': 'User19',
             'folder_type': 'trimmed_data', 'avg': False, 'load_img': True
             },
            ]
'''
Extra (clean) data can also be specified using ld_folders option. This data will be given 
user_id = -1
'''

ld_folders = [{'training_folder': 'Clean_Tremors', 'folder_type': 'positive',
              'data_type': 'tremor', 'load_img': True},

              {'training_folder': 'Clean_Tremors', 'folder_type': 'negative',
              'data_type': 'tremor', 'load_img': True},

              {'training_folder': 'Clean_Earthquakes', 'folder_type': 'positive',
               'data_type': 'earthquake', 'load_img': True, 'process_data': True},

              {'training_folder': 'Clean_Earthquakes', 'folder_type': 'negative',
               'data_type': 'earthquake', 'load_img': True, 'process_data': True}
              ]


ds = QuakeDataSet(ld_files=ld_files, ld_folders=ld_folders, excerpt_len=20000)
print(len(ds.X), len(ds.X_users), len(ds.X_ids), len(ds.X_names), len(ds.Y))

save_file_jl(ds, 'ds_main')
