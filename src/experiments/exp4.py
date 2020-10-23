"""
Exp 4: WavImg (Wavlet Scattering Transform + CNN + FCN) on clean + Earthquake Det Data
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
load_transformed = False
train_mod = False
gen_stats = False


J, Q = 8, 64
excerpt_len = 20000
exp_name = "Exp4_all"

ds = load_file_jl('ds_main')
ds.get_distribution()

if transform_and_save:
    ds = load_file_jl('ds_main')
    ds.get_distribution()
    users = [15, 100]

    ds.modify_flags(avg_flag=True)
    ds.modify_excerpt_len(new_len=20000)

    # Split data into train and test [Gold Users]
    train_indices_gold, test_indices_gold = ds.get_indices_split(train_percent=0.8, seed=True,
                                                                 users=users)
    print("Gold", len(train_indices_gold), len(test_indices_gold))
    ds.modify_flags(avg_flag=True)

    # Get split for other users
    users = [0, 19]
    train_indices, test_indices = ds.get_indices_split(train_percent=0.8, seed=True,
                                                       users=users)

    print("Rest", len(train_indices), len(test_indices))

    train_indices.extend(train_indices_gold)
    print("Total train indices: ", len(train_indices))
    #
    train_set = torch.utils.data.Subset(ds, indices=train_indices)
    test_set = torch.utils.data.Subset(ds, indices=test_indices)
    test_set_gold = torch.utils.data.Subset(ds, indices=test_indices_gold)
    #
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    test_gold_loader = torch.utils.data.DataLoader(test_set_gold, batch_size=batch_size,
                                                   shuffle=True)
    # Transform the data
    combined_train = []
    combined_test = []
    combined_test_gold = []
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
    #
    print("Starting Gold test set: ")
    for index, data in enumerate(test_gold_loader):
        print("Processing batch: {}".format(index))
        coeffs = scatter_transform(data['data'], J=J, Q=Q, excerpt_len=excerpt_len, cuda=True)
        combined_test_gold.append({'data': [coeffs.cpu(), data['img']], 'label': data['label'],
                              'user': data['user'], 'sub_id': data['sub_id'], 'index': data[
                'index']})
        torch.cuda.empty_cache()

    save_file_jl(combined_test_gold, 'combined_test_gold{}_J{}_Q{}'.format(exp_name, J, Q))


if load_transformed:
    # Load file
    transformed_train = load_file_jl('combined_train_{}_J{}_Q{}'.format(exp_name, J, Q))
    transformed_test = load_file_jl('combined_test_{}_J{}_Q{}'.format(exp_name, J, Q))
    transformed_gold = load_file_jl('combined_test_gold{}_J{}_Q{}'.format(exp_name, J, Q))
    print(transformed_train[0]['data'][1].shape)

# Train the model
if train_mod:
    # Initialize the model
    model = ml.models.WavImg(h=200, w=300).to(device)
    """
    All 
    Earthquakes: 2411
    Noise: 1950
    Tremor: 163 
    """
    # Calculated as approx ~ max class examples / specific class examples
    weights_cg = torch.tensor([1, 1.23, 14.79]).to(device)

    loss_func = torch.nn.CrossEntropyLoss(weight=weights_cg)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize Tensorboard
    exp_id = '{}_{}'.format(exp_name, datetime.now().strftime('%d_%m_%Y-%H_%M_%S'))
    writer = SummaryWriter('runs/{}'.format(exp_id))

    train(num_epochs=num_epochs, batch_size=batch_size, model=model, loss_func=loss_func,
          optimizer=optimizer, train_set=transformed_train, test_set=transformed_test,
          test_set_gold=transformed_gold, exp_id=exp_id, writer=writer, save_freq=50, print_freq=20
          , test_freq=20)

if gen_stats:
    ds = load_file_jl('ds_main')
    X_names = ds.X_names
    model = ml.models.WavImg(h=200, w=300).to(device)
    model_name = 'model_Exp4_all_09_10_2020-13_13_44_299.pt'
    model.load_state_dict(torch.load(SAVE_PATH / model_name))

    generate_model_log(model=model, model_name=model_name, sample_set=transformed_train,
                       names=X_names, set='train')

    generate_model_log(model=model, model_name=model_name, sample_set=transformed_test,
                       names=X_names, set='test')



