import data_utils
from collections import defaultdict
from constants import DATA_PATH, META_PATH
from obspy import UTCDateTime, read, read_inventory, Stream
from generate_data import load_station_list
import os
from utils import clean_event_id


def load_traces(folder):
    st = Stream()
    for trace in os.listdir(folder):
        tr = read(str(folder / trace))
        tr[0].stats['split'] = int(trace[-5])  # Retain the split info (split number 1/2 etc)
        st += tr
    return st


if __name__ == '__main__':
    eid = '2012/04/1108:39:31.4'
    # # stats = [['UU', 'SRU', '', 'BHZ,BHN,BHE'], ['TA', 'H17A', '', 'BHZ,BHN,BHE'],
    # #          ['US', 'SDCO', '00', 'BHZ,BH1,BH2']]
    #
    audio_params = {'surface_len': 1000.0, 'damping': 4e-8}
    plot_params = {'surface_len': 1000.0, 'dpi': 100}

    folder = DATA_PATH / 'BSSA' / clean_event_id(eid)
    #
    stats = load_station_list(file_path=DATA_PATH / 'BSSA' / '20120411_station_list.txt')

    # inv = read_inventory(str(folder / 'inv_metadata.xml'), format='STATIONXML')

    # data_utils.load_and_process(event_id=eid, st=None, event_et=3600, inv=inv, stations=stats[
    #                                                                                     :250],
    #                              min_magnitude=8.6, folder_name='Test', save_raw=False,
    #                              split=2, audio_params=audio_params, plot_params=plot_params)

    st = load_traces(folder / 'trimmed_data')
    data_utils.generate_plots_from_trimmed(event_id=eid, st=st, folder_path=folder)