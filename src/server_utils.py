import pandas as pd
import getpass
from panoptes_client import Panoptes, Project, SubjectSet, Subject
from constants import DATA_PATH
import os
from utils import clean_event_id
import datetime


def create_manifest(event_id):
    """
    Create a .csv file mapping different files types of same data
    Args:
        event_id (str): Ideally should be time stamp YYYY-MM-DDTHH:MM:SS.000
    """
    # Load the required files and sort them to ensure correct mapping
    event_id = clean_event_id(event_id)
    audio_files = sorted(os.listdir(DATA_PATH / event_id / 'audio'))
    image_files = sorted(os.listdir(DATA_PATH / event_id / 'plots'))

    # Ensure number of files are same in both directories
    assert len(audio_files) == len(image_files), "Error: No. of audio files differ from no. of " \
                                                 "image files"

    # Convert to dataframe and save
    df = pd.DataFrame(list(zip(image_files, audio_files)), columns=["!image_name", "!audio_name"])
    df.index.name = "subject_id"

    df.to_csv(DATA_PATH / event_id / 'manifest.csv')


def upload_subject_set(event_id):
    """
    Upload the data to Zooniverse platform
    Args:
        event_id (str): Ideally should be time stamp YYYY-MM-DDTHH:MM:SS.000

    """
    event_id = clean_event_id(event_id)
    folder_path = DATA_PATH / event_id
    # Get user credentials
    username = input('Username (Zooniverse):')
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
        set_name = input('Enter subject set name (must be unique)')
        subject_set.display_name = set_name
        subject_set.save()
    elif reply == 2:
        set_id = input("Enter subject set ID")
        subject_set = SubjectSet.find(set_id)

    df = pd.read_csv(folder_path / 'manifest.csv')
    subject_metadata = {}

    # Convert pandas dataframe to dictionary format suitable for subject metadata
    for index, row in df.iterrows():
        file_name = row['!image_name'].split('.')[0]
        subject_metadata[file_name] = {'!audio_name': row['!audio_name'],
                                       '!image_name': row['!image_name'],
                                       '#time_generated': str(datetime.datetime.now())}

    new_subjects = []

    # Upload the files and metadata to Zooniverse
    for filename, metadata in subject_metadata.items():
        subject = Subject()

        subject.links.project = project
        # Give local file path to upload
        subject.add_location(str(folder_path / 'plots' / metadata['!image_name']))
        subject.add_location(str(folder_path / 'audio' / metadata['!audio_name']))

        subject.metadata.update(metadata)
        subject.save()
        new_subjects.append(subject)

    subject_set.add(new_subjects)


if __name__ == '__main__':
    # event_date = "2010_02_27"
    # event_time = "T06_34_13.000"
    # event_id = event_date + event_time
    event_id = "2012_04_11T08_38_37_000"
    create_manifest(event_id)
    upload_subject_set(event_id)
