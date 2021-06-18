# Earthquake Detective 
Earthquake Detective is an online collabarative platform to classify seismograms. This repository connects the platform with a Machine Learning backend for automatic classification and detection of earthquakes and tremors. 
<br>
* Site: https://www.zooniverse.org/projects/vivitang/earthquake-detective
* Paper: https://arxiv.org/abs/2011.04740

## Installation 
* Download [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for your OS 
* Create a new conda environment as follows: 
```
conda create -n quakedet python=3.7 pip
```
* Activate the new conda environment as follows: 
```
conda activate quakedet
```
* Install requirements using the following command: 
```
pip install -r requirements.txt
```
* Install Obspy 
```
conda install obspy
```
* Install cartopy [optional, for plotting 3d plots] 
```
conda install cartopy 
```
* Install PyTorch by following the instructions [here.](https://pytorch.org/get-started/locally/)

## Additional Instructions: 
* For libmagic to correctly work, follow the instructions [here.](https://github.com/ahupp/python-magic#installation)

## Dataset 
* The classification information can be downloaded from [here.]( https://northwestern.box.com/s/jr5y9nw913z2s2artp84vs7f0b7kb8b8) Header format:
```
Subject_ID User_ID Time_Stamp Region_Code Station_Code Channel Label
``` 

* Processed data used in our experiments can be downloaded [here.](https://northwestern.box.com/s/t270hy8vfyr8s0j1zg9lakjyqec799d4)
* Reliabilty dictionary containing reliability scores for the user_ids can be downloaded from [here.](https://northwestern.box.com/s/vmahgb33x89a22l4z7s6yp5o1pdrpvmz)

## Usage 
### Downloading actual data based on the classifications text file 
 * The src/generate_data.py function can be used to download the data with the help of Obspy 
 * To get data from specific users, use the filter_data function with the desired user_ids and optionally the required label. Example: Get data for only User 15: 
 ```
filter_data(path=DATA_PATH / 'classification_data_all_users.txt', user_ids=['15'], name='u15')
 ```
 * After filtering, run load_info_from_labels to get set up the data for downloading. Example: 
```
event_info, user_info = load_info_from_labels(path=DATA_PATH /
                                                    'classification_data_u15.txt')
```
* Finally, call the download data function as follows: 
```
for event_id, stations in event_info.items():
    data_utils.download_data(event_id=event_id, event_et=3600, stations=stations, min_magnitude=7, folder_name='User100', save_raw=False)
```

### Creating dataset from the data 
* Refer to exp0.py for steps to create dataset object from the downloaded data 

### Running the experiments 
* Prepare your data based on the aforementioned steps our download our processed data from [here](https://northwestern.box.com/s/t270hy8vfyr8s0j1zg9lakjyqec799d4) and run experiments as follows: 
```
cd src/experiments 
python experiment{EXP_ID}.py   
```

### Downloading and uploading data to Zooniverse 
* To download data and to get processed seismograms, audio and plots run data_utils.download_data with desired settings 
* To upload data to Zooniverse, run server_utils.py with desired settings 


## File Guide 
### src/ 
* data_utils.py - Contains functions to download and process the data.
* generate_data.py - Use this file to download data over arbitary time windows 
* utils.py - Contains generic utility functions which are used throughout the project 
* server_utils.py - Contains functions to upload data to Zooniverse 
* zooniverse_utils.py - Contains functions to process and analyze the data downloaded from Zooniverse 
* constants.py - Contains constants used throughout the project  
* visualizer.py - Contains functions to visualize seismic data 

### src/ml 
* dataset.py - Contains Pytorch dataset class to load and pre-process data so that it can be used by Pytorch dataloader 
* models.py - Contains code for different ML models used 
* trainer.py - A generic code to train/test different models defined in models.py 
* wavelet.py - Contains function to perform wavelet scattering transform 
* custom_loss.py - Custom loss functions to account for reliability scores 

### src/ml/retirement 
* r_utils.py - Contains functions to calculate reliability scores 
* sample_analysis.py - Contains functions to select and analyze samples based on different selection criteria's 

### src/experiments 
Contains various experiments performed. View experiment files for more details. 


