from constants import DATA_PATH
from obspy import read
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft
from scipy import signal


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


    st2 = read(str(DATA_PATH / "Vivian_Set/2018_09_28T10_02_59_000/processed_data/AK_PTPK__BHZ_2018_09_28T10_02_59_000.sac"))

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
    ft1 = fft(st1[0].data)
    print(ft1.shape)

    f, t, Zxx = signal.stft(st1[0].data, fs=20.0, nperseg=256)
    print(Zxx.shape)
    print(Zxx)
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()