from constants import DATA_PATH
from obspy import read
import matplotlib.pyplot as plt
import numpy as np

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


if __name__ == '__main__':
    st = read(str(DATA_PATH / 'Training_Set_Tremor/Taiwan_20041226/positive/CWB.WHF.EHN.SAC'))
    st2 = read(str(DATA_PATH / 'Training_Set_Tremor/Taiwan_20070912/positive/TW.TPUB.BHN.SAC'))
    st3 = read(str(DATA_PATH / 'Training_Set_Vivian/2014_04_19T13_28_00_000/trimmed_data/AK_GAMB__BHN_2014_04_19T13_28_00_000.sac'))
    # st += st2

    arr = st3[0].data

    arr2 = np.mean(arr.reshape(-1, 4), axis=1)
    print(arr.shape, arr2.shape)

    st3[0].data = 2*np.ones((3, ))
    print(st3[0].data)