"""
This file will contain constants used throughout the project
"""

from pathlib import Path


# Directory paths
PARENT_PATH = Path(__file__).parent
DATA_PATH = PARENT_PATH / '../data'
SAVE_PATH = PARENT_PATH / 'ml/checkpoints'

# Training labels
label_dict = {'Earthquake': 0,
              'Noise': 1,
              'Unclear_Event': 2,
              'Tremor': 3}

