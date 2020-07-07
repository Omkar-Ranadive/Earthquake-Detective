# Earthquake Detective 
Earthquake Detective is an online collabarative platform to classify seismograms. This repository connects the platform with a Machine Learning backend for automatic classification and detection of earthquakes. 

## Installation 
* Download [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for your OS 
* Create a new conda environment as follows: 
```
conda create -n quakedet python=3.7 pip
```
* Install requirements using the following command: 
```
pip install -r requirements.txt
```
* Install cartopy [optional, for plotting 3d plots] 
```
conda install cartopy 
```

## To-do list 
- [ ]  Re-write the data processing functions - Downloading raw data, processing it and generating plots 
- [ ]  Function to convert Seismograms to audio data 
- [ ]  Machine learning classifier for earthquakes vs rest 
- [ ]  Coupling together local traces with larger earthquake traces which could have wave information from larger earthquake traces 
