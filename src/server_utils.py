import pandas as pd
import getpass
from panoptes_client import Panoptes, Project, SubjectSet, Subject
from constants import DATA_PATH
import os
from utils import clean_event_id
from datetime import datetime


def create_manifest(event_id, path, use_filtered=True):
    """
    Create a .csv file mapping different files types of same data
    Args:
        event_id (str): Ideally should be time stamp YYYY-MM-DDTHH:MM:SS.000
        path (str): Path to folder
        use_filtered (bool): If true, plots are loaded from plots_filtered folder
    """
    # Load the required files and sort them to ensure correct mapping
    event_id = clean_event_id(event_id)
    audio_files = sorted(os.listdir(path / event_id / 'audio'))

    if use_filtered:
        image_files = sorted(os.listdir(path / event_id / 'plots_filtered'))
        # If we are using filtered plots, then plots < audio files
        # So only select that subset of audio samples
        trailing_audio_info = audio_files[0][-8:]  # Gets the last part, ex - _400.mp3
        imgs = [img[:-4] for img in image_files]
        audio_files = [img+trailing_audio_info for img in imgs]

    else:
        image_files = sorted(os.listdir(path / event_id / 'plots'))

    # Only upload the BHZ channels
    filtered_audio = [af for af in audio_files if 'BHZ' in af]
    filtered_imgs = [imgf for imgf in image_files if 'BHZ' in imgf]

    # Check if filtered images and audio have equal number of entries
    assert len(filtered_audio) == len(filtered_imgs), "Error: No. of audio files differ from no. of " \
                                                 "image files"

    # Convert to dataframe and save
    df = pd.DataFrame(list(zip(filtered_imgs, filtered_audio)), columns=["!image_name", "!audio_name"])
    df.index.name = "subject_id"
    filename = 'manifest_{}.csv'.format(datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p"))
    df.to_csv(str(path / event_id / filename))


def upload_subject_set(event_id, path, manifest, use_filtered=True):
    """
    NOTE: This function only runs on Python terminal, don't use Pycharm's console (due to
    getpass module)

    Upload the data to Zooniverse platform
    Args:
        event_id (str): Ideally should be time stamp YYYY-MM-DDTHH:MM:SS.000
        path (str): Path to
        manifest (str): The name of manifest (.csv) file created using creat_manifest func
        use_filtered (bool): If true, plots are loaded from the use_filtered folder
    """
    event_id = clean_event_id(event_id)
    folder_path = path / event_id
    plots_folder = 'plots_filtered' if use_filtered else 'plots'
    # Get user credentials
    username = input('Username (Zooniverse): ')
    password = getpass.getpass("Password: ")
    # Connect to Zooniverse
    Panoptes.connect(username=username, password=password)

    # Find project ID
    pID = input('Project ID: ')
    project = Project.find(pID)
    subject_set = SubjectSet()
    subject_set.links.project = pID

    print("Press 1 to create a new subject set")
    print("Press 2 to append to existing subject set")
    reply = int(input())

    if reply == 1:
        set_name = input('Enter subject set name (must be unique): ')
        subject_set.display_name = set_name
        subject_set.save()
    elif reply == 2:
        set_id = input("Enter subject set ID: ")
        subject_set = SubjectSet.find(set_id)

    df = pd.read_csv(folder_path / manifest)
    subject_metadata = {}

    # Convert pandas dataframe to dictionary format suitable for subject metadata
    for index, row in df.iterrows():
        file_name = row['!image_name'].split('.')[0]
        subject_metadata[file_name] = {'!audio_name': row['!audio_name'],
                                       '!image_name': row['!image_name'],
                                       '#time_generated': str(datetime.now())}

    new_subjects = []

    # Upload the files and metadata to Zooniverse
    counter = 0
    for filename, metadata in subject_metadata.items():
        subject = Subject()
        counter += 1

        subject.links.project = project
        # Give local file path to upload
        # NOTE: The order is important here
        # If audio is placed after plots then overlapping image issue occurs
        # So make sure to follow the following order:
        subject.add_location(str(folder_path / 'audio' / metadata['!audio_name']))
        subject.add_location(str(folder_path / plots_folder / metadata['!image_name']))

        subject.metadata.update(metadata)
        subject.save()
        new_subjects.append(subject)

        if counter % 10 == 0:
            print("Processed {} files".format(counter))

    subject_set.add(new_subjects)


if __name__ == '__main__':
    # event_date = "2010_02_27"
    # event_time = "T06_34_13.000"
    # event_id = event_date + event_time
    event_id = "2012/04/1108:39:31.4"
    # create_manifest(event_id=event_id, path=DATA_PATH / 'BSSA')
    #
    upload_subject_set(event_id, path=DATA_PATH / 'BSSA',
                      manifest='manifest_2021_05_28-10_08_22_PM.csv')


