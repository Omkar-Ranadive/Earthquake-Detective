from utils import convert_to_seconds
import data_utils
from collections import defaultdict
from constants import DATA_PATH, META_PATH
from obspy import UTCDateTime


def filter_data(path, user_ids, name="filtered", dtype=[]):
    """
    Filtered data can be generated using this function by specifying user_ids and dtypes. The
    data file containing all users is scanned and a new file is saved containing the filtered
    info.
    Args:
        path (str): Path from which to load the labeled data from
        user_ids (list, type='str'): Provide user ids (second col of data) to only load info from
                                those users
        name (str): File name of the saved file
        dtype (list, type='str'): Provide label info to only load info for those particular labels.
                  Possible Values = 'Earthquake', 'Tremor', 'Noise', 'Unclear Event',
                 'None of the Above'
       """
    file_name = 'classification_data_{}.txt'.format(name)
    with open(DATA_PATH / file_name, 'w') as f1:
        with open(path) as f:
            for line in f.readlines():
                info = line.split()
                if user_ids and info[1] not in user_ids:
                    continue
                if dtype and info[-1] not in dtype:
                    continue
                f1.write(" ".join(info) + '\n')


def load_info_from_labels(path, downloaded_info="", limit=None):
    """
    Function to download data associated with labeled data from Earthquake Detective
    Assumption: Labels are of the form: Sub_Id UserID Time_stamp network station component label
    Args:
        path (str): Path from which to load the labeled data from
        downloaded_info (str): Optionally, provide path to text file which contains subject ids
        Each line of downloaded_info file should only contain the subject id
        of files already downloaded so they can be skipped in the present run
        limit (int): If provided, only samples = limit size will be returned

    Returns: Stations info
    """
    id_stat_dict = defaultdict(list)  # Create a key:val mapping of event_id : stations list

    # Load the list of existing downloaded subject_ids - so that we don't repeat the download
    if downloaded_info:
        files = open(META_PATH / downloaded_info, 'r')
        subject_ids = [sub_id.strip() for sub_id in files.readlines()]
    else:
        subject_ids = []

    counter = 0
    user_data = []
    with open(path) as f:
        for line in f.readlines():
            info = line.split()
            # For now, don't download the none of the above / unclear event data as it is not
            # being classified
            if info[0] not in subject_ids and info[-1] not in ('above', 'Event') and info[5] == \
                    'BHZ':

                if limit and counter + 1 > limit:
                    break

                # Separate event id from station info
                id_stat_dict[info[2]].append([info[3], info[4], "", info[5] + ',BHN,BHE'])
                user_data.append(info)

                counter += 1

    print("Number of samples chosen to download: {}".format(counter))
    return id_stat_dict, user_data


def save_downloaded_info(id_dict, info):
    """
    This function can be used to save subject ids which were recently downloaded.
    Then the saved text file path can be provided to the load_info_from_labels function to skip
    downloading these subject ids if desired.

    The start_date and end_date can be used to only save subject ids within that range
    Useful when only part of the user data is downloaded and we don't want to repeat the
    download in the next run
    Args:
        id_dict (defaultdict): Event info dictionary produced from load_info_from_labels func
        info (list): User info list produced from load_info_from_labels func

    """
    info = sorted(info, key=lambda x: x[2])
    total = 0
    start_date = info[0][2]  # Change this if a different range is required
    end_date = start_date

    for k, v in sorted(id_dict.items()):
        total += len(v)
        if total > 2000:  # This will save the subject ids of the first 2000 samples
            end_date = k
            break
        end_date = k

    print("Total", total)
    with open(META_PATH / 'downloaded_info.txt', 'a') as f:
        for data in info:
            if UTCDateTime(start_date) <= UTCDateTime(data[2]) <= UTCDateTime(end_date):
                    f.write(data[0] + '\n')


def load_station_list(file_path):
    """
    Assumed that each line in file is of the form:
    Net Code, Stat Code, Loc code, Channels
    Args:
        file_path (str): Path to text file containing station info

    Returns (list): Returns List of lists where each list is of the form
                   [Net Code, Stat Code, Loc code, Channels]
    """
    stations = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            info  = line.split()

            if len(info) == 3:
                if info[-1] == 'BHZ':
                    info[-1] += ',BHN,BHE'
                elif info[-1] == 'EHZ':
                    info[-1] += ',EH1,EH2'
                stations.append([info[0], info[1], "", info[2]])
            else:
                if info[-1] == 'BHZ':
                    info[-1] += ',BH1,BH2'
                elif info[-1] == 'EHZ':
                    info[-1] += ',EH1,EH2'
                stations.append([info[0], info[1], info[2], info[3]])

    return stations


if __name__ == '__main__':
    #
    # filter_data(path=DATA_PATH / 'classification_data_all_users.txt', user_ids=['100'],
    #             name='u100')
    #
    # event_info, user_info = load_info_from_labels(path=DATA_PATH /
    #                                                    'classification_data_filtered.txt')

    # print(event_info)
    # for event_id, stations in event_info.items():
    #     data_utils.download_data(event_id=event_id, event_et=3600, stations=stations,
    #                              min_magnitude=7, folder_name='User100', save_raw=False)

    # save_downloaded_sub_ids_info(event_info, user_info)

    """
    Sumatra Earthquake
    """
    eid = '2012/04/1108:39:31.4'
    # # stats = [['UU', 'SRU', '', 'BHZ,BHN,BHE'], ['TA', 'H17A', '', 'BHZ,BHN,BHE'],
    # #          ['US', 'SDCO', '00', 'BHZ,BH1,BH2']]
    #
    audio_params = {'surface_len': 1000.0, 'damping': 4e-8}
    plot_params = {'surface_len': 1000.0, 'dpi': 96}

    stats = load_station_list(file_path=DATA_PATH / 'BSSA' / '20120411_station_list.txt')

    # print(len(stats))
    # print(stats[:250]) # 500:750 downloaded for the full length
    data_utils.download_data(event_id=eid, event_et=3600, stations=stats[:150],
                             min_magnitude=8.6, folder_name='Sumatra', save_raw=False,
                             split=2, audio_params=audio_params, plot_params=plot_params)



    #
    # """
    # Split data testing
    # """
    # #
    # filter_data(path=DATA_PATH / 'classification_data_all_users.txt', user_ids=['0'],
    #             name='u100', dtype=('Tremor'))
    #
    # #
    # #
    # event_info, user_info = load_info_from_labels(path=DATA_PATH /
    #                                                    'classification_data_u100.txt', limit=1)
    #

    #
    # #
    # # """
    # Downloading complete data without any splits
    # """
    #
    # for event_id, stations in event_info.items():
    #     data_utils.download_data(event_id=event_id, event_et=3600, stations=stations,
    #                              min_magnitude=7, folder_name='Tremor_Complete_Samples',
    #                              save_raw=False)

    # # """
    # Downloading data split into two parts
    # """
    # audio_params = {'surface_len': 1000.0, 'damping': 4e-8}
    # plot_params = {'surface_len': 1000.0, 'dpi': 96}
    #
    # for event_id, stations in event_info.items():
    #     data_utils.download_data(event_id=event_id, event_et=3600, stations=stations,
    #                              min_magnitude=7, folder_name='TR_Samples_EZ_s2_d3e',
    #                              save_raw=False,
    #                              split=2, audio_params=audio_params, plot_params=plot_params)


    # """
    # Downloading data split into four parts
    # """
    # audio_params = {'surface_len': 500.0, 'damping': 3e-8}
    # plot_params = {'surface_len': 500.0}
    # #
    # for event_id, stations in event_info.items():
    #     data_utils.download_data(event_id=event_id, event_et=3600, stations=stations,
    #                              min_magnitude=7, folder_name='TR_Samples_EZ_s4_d3e',
    #                              save_raw=False,
    #                              split=4, audio_params=audio_params, plot_params=plot_params)
    #
    #
    #
