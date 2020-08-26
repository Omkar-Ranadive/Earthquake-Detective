# Earthquake Detective 
Earthquake Detective is an online collabarative platform to classify seismograms. This repository connects the platform with a Machine Learning backend for automatic classification and detection of earthquakes. 

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
* Install cartopy [optional, for plotting 3d plots] 
```
conda install cartopy 
```
* Install PyTorch by following the instructions [here.](https://pytorch.org/get-started/locally/)

## Additional Instructions (Windows): 
* For libmagic to correctly work, follow the instructions [here.](https://github.com/ahupp/python-magic#windows)


## Usage 
* To download data and to get processed seismograms, audio and plots run generate_data.py file with desired settings. 
* To upload data to Zooniverse, run server_utils.py with desired settings 

## File Guide 
### src/ 
* data_utils.py - Contains functions to download and process the data.
* generate_data.py - Use this file to download data over arbitary time windows 
* utils.py - Contains generic utility functions which are used throughout the project 
* server_utils.py - Contains function to upload data to Zooniverse 
* constants.py - Contains constants used throughout the project  

### src/ml 
* dataset.py - Contains Pytorch dataset class to load and pre-process data so that it can be used by Pytorch dataloader 
* models.py - Contains code for different ML models used 
* trainer.py - A generic code to train/test different models defined in models.py 
* wavelet.py - Contains function to perform wavelet scattering transform 

### src/experiments 
Contains various experiments performed. View experiment files for more details. 

## To-do list 
- [X]  Re-write the data processing functions - Downloading raw data, processing it and 
generating plots 
- [X]  Function to convert Seismograms to audio data 
- [X] Server side functions to upload to Zooniverse 
- [ ]  Machine learning classifier for earthquakes vs rest 
- [ ]  Coupling together local traces with larger earthquake traces which could have wave 
information from larger earthquake traces 
- [ ]  Explore the possibility of weighing the samples. Example - Stations close to each other 
are more likely to show earthquake at a given time if one of the station's data can be labelled 
as earthquake. Possibly explore record section graphs for this.  


