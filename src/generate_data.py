from utils import convert_to_seconds
import data_utils
from collections import defaultdict
from constants import DATA_PATH, META_PATH
from obspy import UTCDateTime

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
    Assumption: Labels are of the form: Sub_Id User_name Time_stamp network station component label
    Args:
        path (str): Path from which to load the labeled data from

    Returns: Stations info
    """
    id_stat_dict = defaultdict(list)  # Create a key:val mapping of event_id : stations list

    # Load the list of existing downloaded subject_ids - so that we don't repeat the download
    files = open(META_PATH / 'downloaded_files.txt', 'r')
    subject_ids = [sub_id.strip() for sub_id in files.readlines()]
    counter = 0
    user_data = []
    with open(path) as f:
        for line in f.readlines():
            info = line.split()
            # For now, don't download the none of the above / unclear event data as it is not
            # being classified
            if info[0] not in subject_ids and info[-1] not in ('above', 'Event') and info[5] == \
                    'BHZ' and info[-1] == 'Earthquake':
                # Separate event id from station info
                id_stat_dict[info[2]].append([info[3], info[4], "", info[5]+',BHN,BHE'])
                counter += 1
                user_data.append(info)

    print("Number of samples chosen to download: {}".format(counter))
    return id_stat_dict, user_data


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


def get_sub_ids_of_selected_samples(path, id_dict, info):
    info = sorted(info, key=lambda x: x[2])
    total = 0
    start_date = info[0][2]  # Change this if a different range is required
    end_date = start_date

    for k, v in sorted(id_dict.items()):
        total += len(v)
        if total > 2000:
            end_date = k
            break

    print("Total", total)
    for data in info:
        if UTCDateTime(start_date) <= UTCDateTime(data[2]) <= UTCDateTime(end_date):
            with open(META_PATH / 'downloaded_files.txt', 'a') as f:
                f.write(data[0] + '\n')
                # print(data)


def load_golden(path):
    id_stat_dict = defaultdict(list)

    with open(path) as f:
        for line in f.readlines():
            info = line.split()
            id_stat_dict[info[1]].append([info[2], info[3], '', info[4]])

    return id_stat_dict


if __name__ == '__main__':
    """
    Uncomment following section to download data manually by specifying event details 
    """
    # # Specify the download settings
    event_date = "2010_02_27"
    event_time = "T06_34_13.00"
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
    Download data for golden set (or from any text file) 
    
    Refer to extract_info_zooniverse function in utils.py 
    """
    # event_info, _ = load_info_from_labels(path='../data/V_golden.txt')
    #
    #
    '''
    Download for Vivian - Golden User 
    '''
    # event_info, _ = load_info_from_labels(path='../data/classification_data_Vivitang.txt')
    #
    # for event_id, stations in event_info.items():
    #     data_utils.download_data(event_id=event_id, event_et=3600, stations=stations,
    #                              min_magnitude=7, folder_name='Vivian_Set', save_raw=False)

    '''
    Download for Elisabeth - User with huge number of classifications 
    '''
    # event_info, user_info = load_info_from_labels(
    #     path='../data/classification_data_ElisabethB.txt')
    # Download them 1000 at a time
    # total = 0
    # get_sub_ids_of_selected_samples(path='../data/classification_data_ElisabethB.txt',
    #                                 id_dict=event_info, info=user_info)
    '''
    Download for Jeff - Classifies Earthquake with good accuracy - only download earthquake 
    samples 
    '''
    event_info, user_info = load_info_from_labels(
        path='../data/classification_data_Jeff503.txt')

    for event_id, stations in event_info.items():
        data_utils.download_data(event_id=event_id, event_et=3600, stations=stations,
                                 min_magnitude=7, folder_name='Jeff_Set', save_raw=False)

    # for event_id, stations in sorted(event_info.items()):
    #     total += len(stations)
    #     # if total >= 1500:
    #     #     data_utils.download_data(event_id=event_id, event_et=3600, stations=stations,
    #     #                          min_magnitude=7, folder_name='ElisabethB_set', save_raw=False)
    #     print(event_id, total)
    #
    #     if total >= 2000:
    #         break


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

    """
    Download data for golden set classification
    """
    # event_info = load_golden(path=DATA_PATH / 'Golden' / 'golden_to_be_classified.txt')
    #
    # for event_id, stations in event_info.items():
    #     data_utils.download_data(event_id=event_id, event_et=3600, stations=stations,
    #                              min_magnitude=7, folder_name="Golden_Samples")
