"""
Exp 5:
Model run for BSSA Paper
on Origin times and hypocenters of 38 earthquakes of Mw >= 7.0,
along with whether their surface waves potentially triggered tremor in Yellowstone (H17A) or
local earthquakes in central Utah (SRU) or the Raton Basin (SDCO).
"""

import sys
sys.path.append('../../src/')
import ml.models
from ml.dataset import QuakeDataSet
import torch
from torch.utils.tensorboard import SummaryWriter
from ml.trainer import train, test, generate_model_log
from ml.wavelet import scatter_transform
from constants import SAVE_PATH, DATA_PATH
from utils import save_file_jl, load_file_jl
from datetime import datetime


ld_unlabeled = [{'folder_path': DATA_PATH / 'BSSA' / '2012_04_1108_39_31_4',
                 'load_img': True, 'process_data': False, 'resample': True},]


ds = QuakeDataSet(ld_files=None, ld_folders=None, ld_unlabeled=ld_unlabeled, excerpt_len=20000)
print(len(ds.X), len(ds.X_users), len(ds.X_ids), len(ds.X_names), len(ds.Y))
# save_file_jl(ds, 'ds_main')

data_loader = torch.utils.data.DataLoader(ds, batch_size=len(ds))


exp_name = "exp5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
J, Q = 8, 64
excerpt_len = 20000

transform_and_save = True
load_transformed = True
gen_stats = True


if transform_and_save:
    transformed_data = []
    for index, data in enumerate(data_loader):
        print("Processing batch: {}".format(index))
        coeffs = scatter_transform(data['data'], J=J, Q=Q, excerpt_len=excerpt_len, cuda=True)
        # Make sure to pass both seismic and coeffs as we will be training on both
        transformed_data.append({'data': [coeffs.cpu(), data['img']], 'label': data['label'],
                               'user': data['user'], 'sub_id': data['sub_id'], 'index':
                                   data['index']})
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary(0))

    save_file_jl(transformed_data, 'transformed_data_{}_J{}_Q{}'.format(exp_name, J, Q))


if gen_stats:
    X_names = ds.X_names
    model = ml.models.WavImg(h=200, w=300).to(device)
    # model_name = 'model_Exp4_all_09_10_2020-13_13_44_299.pt'
    model_name = 'model_Exp3_CG_Data_09_10_2020-01_05_57_299.pt'
    model.load_state_dict(torch.load(SAVE_PATH / model_name))

    generate_model_log(model=model, model_name=model_name, sample_set=transformed_data,
                       names=X_names, set='unlabeled')



