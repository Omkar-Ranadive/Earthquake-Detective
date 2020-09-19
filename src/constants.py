"""
This file will contain constants used throughout the project
"""

from pathlib import Path


# Directory paths
PARENT_PATH = Path(__file__).parent
DATA_PATH = PARENT_PATH / '../data'
SAVE_PATH = PARENT_PATH / 'ml/checkpoints'
META_PATH = PARENT_PATH / '../meta'

# Training labels
label_dict = {'Earthquake': 0,
              'Noise': 1,
              'Tremor': 2,
              'Unclear_Event': 3,
              'Unclear Event': 3,
              'Event': 3,
              'None of the Above': 3,
              'None of the above': 3,
              'above': 3
              }

folder_labels = {'earthquake': {'positive': 0, 'negative': 1},
                 'tremor': {'positive': 2, 'negative': 1}}

# Number of classes (Currently, only considering Earthquake, Noise and Tremor)
n_classes = 3


