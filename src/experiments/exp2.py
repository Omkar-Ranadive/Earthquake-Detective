"""
Exp 2: WavNet (Wavlet Scattering Transform + Fully Connected NN) on clean + gold users data
user_ids = [-1, 100, 15]
"""

import sys
sys.path.append('../../src/')
import ml.models
import torch
from torch.utils.tensorboard import SummaryWriter
from ml.trainer import train, test, generate_model_log
from ml.wavelet import scatter_transform
from constants import SAVE_PATH, META_PATH
from utils import save_file_jl, load_file_jl
from datetime import datetime


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper parameters
num_epochs = 300
batch_size = 100
learning_rate = 1e-5


transform_and_save = False
load_transformed = True
train_mod = False
gen_stats = True


J, Q = 8, 64
excerpt_len = 20000
exp_name = "Exp2_CG_Data"


if transform_and_save:
    ds = load_file_jl('ds_main')
    ds.get_distribution()
    users = [15, 100]
    train_indices, test_indices = ds.get_indices_split(train_percent=0.8, seed=False, users=users)
    print(len(train_indices), len(test_indices))

    ds.modify_flags(avg_flag=True)
    ds.modify_excerpt_len(new_len=20000)

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
                               'user': data['user'], 'sub_id': data['sub_id'], 'index':
                                   data['index']})
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary(0))

    save_file_jl(combined_train, 'combined_train_{}_J{}_Q{}'.format(exp_name, J, Q))

    print("Starting test set: ")
    for index, data in enumerate(test_loader):
        print("Processing batch: {}".format(index))
        coeffs = scatter_transform(data['data'], J=J, Q=Q, excerpt_len=excerpt_len, cuda=True)
        combined_test.append({'data': [coeffs.cpu(), data['img']], 'label': data['label'],
                              'user': data['user'], 'sub_id': data['sub_id'], 'index': data[
                'index']})
        torch.cuda.empty_cache()

    save_file_jl(combined_test, 'combined_test_{}_J{}_Q{}'.format(exp_name, J, Q))


if load_transformed:
    # Load file
    transformed_train = load_file_jl('combined_train_{}_J{}_Q{}'.format(exp_name, J, Q))
    transformed_test = load_file_jl('combined_test_{}_J{}_Q{}'.format(exp_name, J, Q))
    print(transformed_train[0]['data'][1].shape)

# Train the model
if train_mod:
    # Initialize the model
    model = ml.models.WavNet().to(device)
    """
    Clean samples + gold (train) 
    Earthquakes: 825 
    Noise: 812 
    Tremor: 39  
    """
    # Calculated as approx ~ max class examples / specific class examples
    weights_cg = torch.tensor([1, 1.02, 20]).to(device)

    loss_func = torch.nn.CrossEntropyLoss(weight=weights_cg)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize Tensorboard
    exp_id = '{}_{}'.format(exp_name, datetime.now().strftime('%d_%m_%Y-%H_%M_%S'))
    writer = SummaryWriter('runs/{}'.format(exp_id))

    train(num_epochs=num_epochs, batch_size=batch_size, model=model, loss_func=loss_func,
          optimizer=optimizer, train_set=transformed_train, test_set=transformed_test,
          exp_id=exp_id, writer=writer, save_freq=50, print_freq=20, test_freq=20)

if gen_stats:
    ds = load_file_jl('ds_main')
    X_names = ds.X_names
    model = ml.models.WavNet().to(device)
    weights_c = torch.tensor([1.03, 1, 14.25]).to(device)
    loss_func = torch.nn.CrossEntropyLoss(weight=weights_c)
    model_name = 'model_Exp2_CG_Data_09_10_2020-00_53_32_299.pt'
    model.load_state_dict(torch.load(SAVE_PATH / model_name))

    generate_model_log(model=model, model_name=model_name, sample_set=transformed_train,
                       names=X_names, set='train')

    generate_model_log(model=model, model_name=model_name, sample_set=transformed_test,
                       names=X_names, set='test')



