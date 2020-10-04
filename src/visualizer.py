from constants import DATA_PATH
from obspy import read
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy import signal
from PIL import Image


def plot_multiple(st):
    num_traces = len(st)
    rows = num_traces//2
    cols = num_traces
    fig, ax = plt.subplots(rows, cols)

    if rows == 1:
        for c, tr in enumerate(st):
            ax[c].set_title("-".join((tr.stats.network, tr.stats.station, tr.stats.channel)))
            ax[c].plot(tr.times("matplotlib"), tr.data, "b-")
            ax[c].xaxis_date()
    else:
        for r in range(rows):
            for c, tr in enumerate(st):
                ax[r, c].set_title("-".join((tr.stats.network, tr.stats.station,
                                             tr.stats.channel)))
                ax[r, c].plot(tr.times("matplotlib"), tr.data, "b-")
                ax[r, c].xaxis_date()

    fig.autofmt_xdate()
    plt.show()


def _resample_data(st, desired_rate=20.0):
    # Set no filter to False to perform automatic anti aliasing
    st.resample(sampling_rate=desired_rate, no_filter=False)
    return st


if __name__ == '__main__':
    st1 = read(str(DATA_PATH / "Vivian_Set/2012_04_11T08_38_37_000/trimmed_data"
                               "/AK_BESE__BHE_2012_04_11T08_38_37_000.sac"))

    st1_plot = Image.open(str(DATA_PATH / "Vivian_Set/2012_04_11T08_38_37_000/plots"
                               "/AK_BESE__BHE_2012_04_11T08_38_37_000.png"))
    st2 = read(str(DATA_PATH / "Vivian_Set/2018_09_28T10_02_59_000/processed_data/AK_PTPK__BHZ_2018_09_28T10_02_59_000.sac"))

    st2_plot = Image.open(str(DATA_PATH / "Vivian_Set/2018_09_28T10_02_59_000/plots"
                                 "/AK_PTPK__BHZ_2018_09_28T10_02_59_000.png"))

    rate, aud1 = wavfile.read(str(DATA_PATH /
                             "Vivian_Set/2012_04_11T08_38_37_000/audio/AK_BESE__BHE_2012_04_11T08_38_37_000_400.mp3"))

    print(st1[0].data.shape)

    # st2[0].plot()
    print(rate, aud1.shape, aud1.shape[0]/rate)
    length = aud1.shape[0] / rate
    # time = np.linspace(0., length, aud1.shape[0])
    # plt.plot(time, aud1)
    # plt.legend()
    # plt.xlabel("Time [s]")
    # plt.ylabel("Amplitude")
    # plt.show()
    import torch
    # fft = np.fft.fft(st2[0].data)
    # print(fft.shape)
    # fft = np.abs(fft)
    # plt.plot(fft)
    # plt.show()

    # stft_trans = torch.stft(torch.tensor(st1[0].data), n_fft=1024)
    # print(stft_trans.shape)
    # stft_trans = torch.norm(stft_trans, dim=-1)
    # print(stft_trans.shape)
    # print(stft_trans)
    #
    #
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # fig, ax = plt.subplots()
    # ax.axis('off')
    # print(st1_plot.mode, st1_plot.format)
    # st1_plot = st1_plot.convert(mode="1", dither=Image.NONE)
    # print(st1_plot.mode, st1_plot.format)
    # width, height = st1_plot.size
    # new_img = st1_plot.crop((80, 0, width, height-275))
    # new_img = new_img.resize((320, 200))
    # new_img = np.asarray(new_img)
    # print(new_img.shape)

    print("Original", st1_plot.mode)
    st1_plot = st1_plot.convert(mode="1", dither=Image.NONE)
    print("After converting: ", st1_plot.mode)
    width, height = st1_plot.size
    new_img = st1_plot.crop((80, 0, width, height - 275))
    print("After cropping: ", new_img.mode)
    new_img = new_img.resize((320, 200))
    print("After resizing: ", new_img.mode)
    new_img = np.asarray(new_img)
    print("After numpy array: ", new_img.shape)


    # # new_img = st1_plot[:-270, :, :]
    # # new_img2 = st2_plot[:-270, :, :]
    # # print(new_img.shape)
    # # new_img = new_img.re
    # fig, ax = plt.subplots()
    # ax.imshow(new_img, cmap='gray')
    # ax.axis('off')
    #
    # plt.show()
    #
    # st_clean = read(str(DATA_PATH / 'Training_Set_Prem/SAC_20041226_2_XF_prem/positive/H1000.BHZ'
    #                               '.SAC'))
    # tr = st_clean[0]
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot(tr.times("matplotlib"), tr.data, "b-")
    # ax.axis('off')
    # plt.savefig('test.png', dpi=300, bbox_inches='tight', pad_inches=0)
    # plt.show()