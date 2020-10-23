from constants import DATA_PATH
from obspy import read
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy import signal
from PIL import Image
import os
import librosa
import librosa.display
import seaborn as sns
import pandas as pd
from scipy.signal import savgol_filter


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def process_data(st):
    st.detrend('demean')
    st.filter('bandpass', freqmin=2, freqmax=8, corners=4, zerophase=True)
    return st


def plot_three_channel_data(st1, st2, st3, pd=False):
    st = st1 + st2 + st3
    if pd:
        st = process_data(st)
    st.plot()


def plot_runs(file_names, path, save_name, labels, smooth=True):
    sns.set(font_scale=1.2)
    sns.set_style("whitegrid")
    combined_data = pd.DataFrame(columns=['Wall time', 'Step', 'Value', 'Number'])
    for i, file_name in enumerate(file_names):
        exp = pd.read_csv(path / file_name)
        if smooth:
            exp['Value'] = savgol_filter(exp['Value'], 9, 3)
        exp['Number'] = i
        combined_data = pd.concat([combined_data, exp])

    # sns_plot = sns.relplot(x="Step", y="Value", data=combined_data, kind="line", hue="Number")
    # sns_plot.set(xlabel='Epochs', ylabel='Accuracy')
    #
    plt.figure(figsize=(7, 5))
    sns.lineplot(x="Step", y="Value", data=combined_data, hue="Number", legend=False,
                 palette=sns.color_palette('bright', n_colors=len(file_names))).set_title('Test')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.legend(labels, loc='lower right')
    plt.tight_layout()
    plt.savefig(path / save_name, dpi=300)


if __name__ == '__main__':
    # file_names = ['exp1.csv', 'exp2.csv', 'exp3.csv', 'exp4.csv']
    # labels = ['Wavnet C', 'Wavnet CG', 'WavImg CG', 'WavImg All']
    # plot_runs(file_names=file_names, path=DATA_PATH / 'Paper', save_name='acc_comp',
    #           labels=labels, smooth=False)

    #
    file_names = ['exp1_t.csv', 'exp2_t.csv', 'exp3_t.csv', 'exp4_t.csv', 'exp4_tg.csv']
    labels = ['Wavnet C', 'Wavnet CG', 'WavImg CG', 'WavImg All', 'WavImg_Gold']
    plot_runs(file_names=file_names, path=DATA_PATH / 'Paper', save_name='test_acc_comp',
              labels=labels)

