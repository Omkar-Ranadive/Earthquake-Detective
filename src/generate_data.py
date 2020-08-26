from utils import convert_to_seconds
import data_utils
from collections import defaultdict


def load_stations(path):
    """
    Load station data in a list
    Args:
        path (str): Path to the file to load station codes from

    Returns: Stations in comma separated format (compatible with Obspy)
    """

    stations = []
    with open(path) as f:
        for stat in f.readlines():
            stations.append(stat.strip())

    final_list = ",".join(stations)
    return final_list


def load_info_from_labels(path):
    """
    Function to download data associated with labeled data from Earthquake Detective
    Assumption: Labels are of the form: Time_stamp network station component label
    Args:
        path (str): Path from which to load the labeled data from

    Returns: Stations info
    """
    id_stat_dict = defaultdict(list)  # Create a key:val mapping of event_id : stations list
    with open(path) as f:
        for line in f.readlines():
            info = line.split()
            # Separate event id from station info
            id_stat_dict[info[0]].append([info[1], info[2], "", info[3]+',BHN,BHE'])

    return id_stat_dict


def load_from_catalog(path):

    id_stat_dict = defaultdict(list)
    with open(path) as f:
        for line in f.readlines():
            info = line.split()
            if info and '#' not in info[0] and info[2] == 'eq':
                # Format of data:
                # Date, Time, Label, Other info......
                event_id = info[0].replace('/', '_') + 'T' + info[1]
                id_stat_dict[event_id].append(['CI', 'WOR', '', 'BHZ,BHN,BHE'])

    return id_stat_dict


if __name__ == '__main__':
    """
    Uncomment following section to download data manually by specifying event details 
    """
    # # Specify the download settings
    event_date = "2010_02_27"
    event_time = "T06_34_13.000"
    event_id = event_date + event_time
    # event_et = convert_to_seconds(3600, t='s')
    # stations = [['AK', 'SAW', '', 'BHZ,BHN,BHE']]
    # min_mag = 7
    #
    # # final_list = load_stations('../meta/Alaska_Stations.txt')
    #
    # # stations = [['AK', final_list, '', 'BHZ,BHN,BHE']]
    # # Call the download function
    # data_utils.download_data(event_id=event_id, event_et=event_et, stations=stations,
    #                          min_magnitude=min_mag)

    """
    Download data for golden set 
    """
    # event_info = load_info_from_labels(path='../data/V_golden.txt')
    #
    # for event_id, stations in event_info.items():
    #     data_utils.download_data(event_id=event_id, event_et=3600, stations=stations,
    #                              min_magnitude=7)

    """
    Download data from catalog files 
    """
    # event_info = load_from_catalog('../data/Training_Set_California/SCEC_DC/1999.catalog')
    # print(event_info)


    # st = data_utils.download_data_direct(event_id=event_id, stations=[['CI', 'PASC', '--',
    #                                                                                    'BLE']])
    # print(st)
    #
    # st.plot()
    # for event_id, stations in event_info.items():
    #     data_utils.download_data(event_id=event_id, event_et=3600, stations=stations,
    #                              min_magnitude=7)
