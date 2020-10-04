import sys
sys.path.append('../../src/')
import ml.models
import torch
from ml.dataset import QuakeDataSet
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
from ml.trainer import train, test
from constants import SAVE_PATH
from utils import save_file_jl, load_file_jl
from ml.wavelet import scatter_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper parameters
num_epochs = 150
batch_size = 100
learning_rate = 1e-5

# Load the data as PyTorch tensors
ld_files = [{'file_name': '../../data/classification_data_Vivitang.txt',
             'training_folder': 'Vivian_Set',
             'folder_type': 'trimmed_data', 'avg': False, 'load_img': True},

            {'file_name': '../../data/classification_data_suzanv.txt',
             'training_folder': 'Testing_Set_Suzan',
            'folder_type': 'trimmed_data', 'avg': False, 'load_img': True
             },

            {'file_name': '../../data/classification_data_ElisabethB.txt',
             'training_folder': 'ElisabethB_set',
             'folder_type': 'trimmed_data', 'avg': False, 'load_img': True
             },

            {'file_name': '../../data/classification_data_Jeff503.txt',
             'training_folder': 'Jeff_Set',
             'folder_type': 'trimmed_data', 'avg': False, 'load_img': True
             },
            ]


ld_folders = [{'training_folder': 'Training_Set_Tremor', 'folder_type': 'positive',
              'data_type': 'tremor', 'load_img': True},

              {'training_folder': 'Training_Set_Prem', 'folder_type': 'positive',
               'data_type': 'earthquake', 'load_img': True},

              {'training_folder': 'Training_Set_Prem', 'folder_type': 'negative',
               'data_type': 'earthquake', 'load_img': True}
              ]


ds = QuakeDataSet(ld_files=ld_files, ld_folders=ld_folders, excerpt_len=20000)
print(len(ds.X), len(ds.X_users), len(ds.X_ids), len(ds.X_names), len(ds.Y))
# save_file_jl(ds, 'ds_dynamic_imgs_2')

ds = load_file_jl('ds_dynamic_imgs')
ds.get_distribution()

transform_and_save = False
load_transformed = False
train_mod = False
gen_stats = False

J, Q = 8, 64
excerpt_len = 20000
exp_name = "Exp9"

if transform_and_save:
    users = ['Vivitang', 'suzanv']
    # Split data into train and test
    train_indices, test_indices = ds.get_indices_split(train_percent=0.8, seed=False, users=users)
    print(len(train_indices), len(test_indices))
    ds.modify_flags(avg_flag=True)

    train_set = torch.utils.data.Subset(ds, indices=train_indices)
    test_set = torch.utils.data.Subset(ds, indices=test_indices)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # Transform the data
    combined_train = []
    combined_test = []
    for index, data in enumerate(train_loader):
        print("Processing batch: {}".format(index))
        coeffs = scatter_transform(data['data'], J=J, Q=Q, excerpt_len=excerpt_len, cuda=True)
        # Make sure to pass both seismic and coeffs as we will be training on both
        combined_train.append({'data': [coeffs.cpu(), data['img']], 'label': data['label'],
                               'user': data['user'], 'sub_id': data['sub_id']})
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary(0))

    save_file_jl(combined_train, 'combined_train_{}_J{}_Q{}'.format(exp_name, J, Q))

    print("Starting test set: ")
    for index, data in enumerate(test_loader):
        print("Processing batch: {}".format(index))
        coeffs = scatter_transform(data['data'], J=J, Q=Q, excerpt_len=excerpt_len, cuda=True)
        combined_test.append({'data': [coeffs.cpu(), data['img']], 'label': data['label'],
                              'user': data['user'], 'sub_id': data['sub_id']})
        torch.cuda.empty_cache()

    save_file_jl(combined_test, 'combined_test_{}_J{}_Q{}'.format(exp_name, J, Q))

if load_transformed:
    # Load file
    transformed_train = load_file_jl('combined_train_{}_J{}_Q{}'.format(exp_name, J, Q))
    transformed_test = load_file_jl('combined_test_{}_J{}_Q{}'.format(exp_name, J, Q))
    print(transformed_train[0]['data'][0].shape)


# Train the model
if train_mod:
    # Initialize the model
    # model = ml.models.WavImg(h=200, w=300).to(device)

    # Image net baseline
    model = ml.models.ImgNet(h=200, w=300).to(device)

    # Load existing model to continue training
    # model_name = 'model_Exp5_29_09_2020-15_26_09_199.pt'
    # model.load_state_dict(torch.load(SAVE_PATH / model_name))
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize Tensorboard
    exp_id = '{}_{}'.format(exp_name, datetime.now().strftime('%d_%m_%Y-%H_%M_%S'))
    writer = SummaryWriter('runs/{}'.format(exp_id))

    train(num_epochs=num_epochs, batch_size=batch_size, model=model, loss_func=loss_func,
          optimizer=optimizer, train_set=transformed_train, test_set=transformed_test,
          exp_id=exp_id, writer=writer,  save_freq=50, print_freq=20, test_freq=20)

