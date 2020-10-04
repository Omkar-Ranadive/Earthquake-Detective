from obspy.clients.fdsn import Client, RoutingClient
from obspy import UTCDateTime, Inventory, Stream
from obspy.clients.fdsn.header import FDSNException
from obspy.geodetics.base import gps2dist_azimuth
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
import os
import constants
from scipy.io import wavfile
import numpy as np
from utils import clean_event_id
# This parameter will prevent matplotlib from throwing errors for plots with large data points
mpl.rcParams['agg.path.chunksize'] = 10000


def create_folders(event_id, folder_name="default_folder"):
    """
    Args:
        event_id (str): Ideally should be time stamp YYYY-MM-DDTHH:MM:SS.000
        folder_name (str): Name of the folder in which the data gets saved

    Returns:
        folder_path (Path obj): Returns folder path so that other functions can save in correct
        location
    """

    # Replace ':' with '_' as it that is an illegal directory char on Windows
    event_id = clean_event_id(event_id)

    # Create the required folders
    if not os.path.exists(constants.DATA_PATH / folder_name):
        os.mkdir(constants.DATA_PATH / folder_name)

    folder_path = constants.DATA_PATH / folder_name / event_id
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    if not os.path.exists(folder_path / 'raw_data'):
        os.mkdir(folder_path / 'raw_data')
    if not os.path.exists(folder_path / 'processed_data'):
        os.mkdir(folder_path / 'processed_data')
    if not os.path.exists(folder_path / 'plots'):
        os.mkdir(folder_path / "plots")
    if not os.path.exists(folder_path / 'audio'):
        os.mkdir(folder_path / 'audio')
    if not os.path.exists(folder_path / 'trimmed_data'):
        os.mkdir(folder_path / 'trimmed_data')

    return folder_path


def download_data(event_id, stations, min_magnitude=7, event_client="USGS",
                  event_et=3600, stat_client="IRIS", save_raw=True,
                  process_d=True, sampling_rate=40.0, gen_plot=True, gen_audio=True,
                  folder_name="default_folder"):
    """
    Download and save the raw data
    Args:
        event_id: Ideally should be time stamp YYYY-MM-DDTHH:MM:SS.000
        stations (list of lists):  Each entry of the list is a list of form [network, station,
                                   location, channel]
        min_magnitude (float): Events >= min_magnitude will be retrieved
        event_client (str): Client to use for retrieving events
        event_et (int): Specifies how long the event range should be from start time (in seconds)
        stat_client (str): Client to use for retrieving waveforms and inventory info
        save_raw (bool): If true, raw data will also be saved (takes a lot of space)
        process_d (bool): If true, the raw data is also processed
        sampling_rate (int): Sampling rate in Hz
        gen_plot (bool): If true, plots are generated
        gen_audio (bool): If true, audio is generated from the waveforms
        folder_name (str): Name of the folder in which the data gets saved

    """
    # Make sure the folders are created
    folder_path = create_folders(event_id, folder_name=folder_name)

    # Download event information with mag >= min_magnitude
    e_client = Client(event_client)
    event_id_utc = UTCDateTime(event_id)
    event_cat = e_client.get_events(starttime=event_id_utc - 100, endtime=event_id_utc + event_et,
                                    minmagnitude=min_magnitude)
    # For each event, find local earthquakes
    s_client = Client(stat_client)
    print(event_cat)
    for event in event_cat:
        origin = event.preferred_origin()
        start_time = origin.time
        end_time = origin.time + 2 * 3600
        inv = Inventory(networks=[], source="")

        # Download inventory info based on the station client and station names
        for net, sta, loc, cha in stations:
            try:
                inv += s_client.get_stations(network=net, station=sta, location=loc,
                                             channel=cha, level="response")
            except FDSNException:
                print("Failed to download inventory information {} {}".format(net, sta))
        print("Inventory downloaded")
        # Get the Seismograms
        st = Stream()
        for net, sta, loc, cha in stations:
            try:
                st += s_client.get_waveforms(network=net, station=sta, location=loc,
                                             channel=cha, starttime=start_time, endtime=end_time,
                                             attach_response=True)
            except FDSNException:
                print("Failed to download waveform information for {} {}".format(net, sta))
        print("Seismograms trimmed")
        # Check for split-streams and remove them
        # These are basically corrupt streams with discontinuities between them
        for tr in st:
            if abs(start_time - tr.stats.starttime) > 1 or abs(end_time - tr.stats.endtime) > 1:
                st.remove(tr)

        #  Save the raw data
        raw_path = folder_path / 'raw_data'

        # Replace '.' and '-' in event_id before saving
        event_id_new = clean_event_id(event_id)

        # Save the files
        if save_raw:
            for tr in st:
                file_id = "_".join((tr.stats.network, tr.stats.station, tr.stats.location,
                                   tr.stats.channel, event_id_new))
                file_path = raw_path / (file_id + ".sac")
                tr.write(str(file_path), format='SAC')

        # Process the seismograms if process_d flag = True
        if process_d:
            st = process_data(event_id=event_id, st=st, sampling_rate=sampling_rate,
                              folder_name=folder_name)

        if gen_plot:
            generate_plots(event_id=event_id, st=st, origin=origin, inv=inv,
                           sampling_rate=sampling_rate,  folder_name=folder_name)

        if gen_audio:
            generate_audio(event_id=event_id, st=st, origin=origin, inv=inv,
                           sampling_rate=sampling_rate,  folder_name=folder_name)


def download_data_direct(event_id, stations, min_magnitude=2.5, event_client="SCEDC",
                         event_et=2000,
                  process_d=True, sampling_rate=40.0, gen_plot=True, gen_audio=True):
    """
    Download data directly given the information. Use this function to download data from the
    catalog directly. I.E We aren't searching for local earthquakes corresponding to a large
    earthquake.
    Args:
      event_id: Ideally should be time stamp YYYY-MM-DDTHH:MM:SS.000
      stations (list of lists):  Each entry of the list is a list of form [network, station,
                                 location, channel]
      min_magnitude (float): Events >= min_magnitude will be retrieved
      event_client (str): Client to use for retrieving events
      event_et (int): Specifies how long the event range should be from start time (in seconds)
      process_d (bool): If true, the raw data is also processed
      sampling_rate (int): Sampling rate in Hz
      gen_plot (bool): If true, plots are generated
      gen_audio (bool): If true, audio is generated from the waveforms

    """

    # NOTE: Incomplete function - do not use
    client = Client(event_client)
    # client = RoutingClient("iris-federator")
    utc = UTCDateTime(event_id)
    print("UTC: ", utc)
    starttime = utc - 100
    st = Stream()

    for net, sta, loc, cha in stations:
        try:
            st += client.get_waveforms(network=net, station=sta, location=loc,
                                         channel=cha, starttime=starttime, endtime=starttime + event_et)
        except FDSNException:
            print("Failed to download inventory information {} {}".format(net, sta))

    return st


def process_data(event_id, st, sampling_rate, pre_filt=(1.2, 2, 8, 10), water_level=100,
                 folder_name="default_folder"):
    """
    Function to process the raw data stream
    Args:
        event_id (str): Event ID -> Ideally should be time stamp YYYY-MM-DDTHH:MM:SS.000
        st (obspy stream obj): Stream object which includes traces
        sampling_rate (float): Sampling rate in Hertz
        pre_filt (tuple): Four corner frequencies for applying bandpass filter
        water_level (int): Water level for de-convolution
        folder_name (str): Name of the folder in which the data gets saved


    Returns:
        st (Obspy Stream obj): Returns the processed stream object
    """

    # Make sure the right folders are created
    folder_path = create_folders(event_id,  folder_name=folder_name)

    # Re-sample and de-trend the stream data
    st.resample(sampling_rate=sampling_rate, window='hanning', no_filter=True, strict_length=False)
    st.detrend('demean')  # Center the data around the mean

    # Remove instrument response -> Y axis gets changed to velocity
    st.remove_response(output='VEL', pre_filt=pre_filt, zero_mean=False, taper=False,
                       water_level=water_level)

    # Replace '.' and '-' and ':' in event_id before saving
    event_id = clean_event_id(event_id)

    # Save the files
    processed_path = folder_path / 'processed_data'

    for tr in st:
        file_id = "_".join((tr.stats.network, tr.stats.station, tr.stats.location,
                            tr.stats.channel, event_id))
        file_path = processed_path / (file_id + ".sac")
        tr.write(str(file_path), format='SAC')

    return st


def generate_plots(event_id, st, origin, inv, sampling_rate, group_vel=4.5, surface_len=2000.0,
                   folder_name="default_folder"):
    """
    Generate trimmed plots of the seismograms
    Args:
        event_id (str): Event ID -> Ideally should be time stamp YYYY-MM-DDTHH:MM:SS.000
        st (Obspy stream obj): Stream object which includes traces
        origin (Obspy origin obj): Origin object obtained for each event
        inv (Obspy inventory obj): Inventory information obtained using get_stations() func
        sampling_rate (float): Sampling rate in Hertz
        group_vel (float): Velocity at which waves move in km/s (Default=4.5; known value of
                           surface waves)
        surface_len (float): Surface window length in seconds which determines how long the
                             trimmed trace will be
        folder_name (str): Name of the folder in which the data gets saved
    """
    # Make sure the right folders are created
    folder_path = create_folders(event_id,  folder_name=folder_name)

    # Trim Seismograms
    for tr in st:
        coordinates = inv.get_coordinates(".".join((tr.stats.network, tr.stats.station,
                                                    tr.stats.location, tr.stats.channel)),
                                          datetime=origin.time)

        sta_lat, sta_lon = coordinates['latitude'], coordinates['longitude']
        # gps2dist returns distance (in meters) and forward, backward azimuths. Select the distance
        dist_origin_to_sta = gps2dist_azimuth(origin.latitude, origin.longitude, sta_lat,
                                              sta_lon)[0]

        # Dist / 1000 converts it to distance in km.
        # Then dividing it by group vel (km/s) converts it to seconds
        start = origin.time + (dist_origin_to_sta/1000)/group_vel
        end = start + surface_len
        tr.trim(starttime=start, endtime=end, pad=True, nearest_sample=False, fill_value=0)

    # Replace '.' and '-' in event_id before saving
    event_id = clean_event_id(event_id)

    # Save the trimmed files
    trimmed_path = folder_path / 'trimmed_data'

    for tr in st:
        file_id = "_".join((tr.stats.network, tr.stats.station, tr.stats.location,
                            tr.stats.channel, event_id))
        file_path = trimmed_path / (file_id + ".sac")
        tr.write(str(file_path), format='SAC')

    # Create the plots using matplotlib
    fig, ax = plt.subplots()

    for tr in st:
        # Tr.times('matplotlib') returns time in number of days since day 0001 (i.e 01/01/01 AD)
        x_coordinates = tr.times('matplotlib')

        ax.plot(tr.times('matplotlib'), tr.data, c='b')
        # Format x axis in readable data format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis_date()
        # Limit the y axis to known local earthquake range.
        ax.set_ylim(-1e-7, 1e-7)
        # Trim any unnecessary space around x axis by limiting to its range
        ax.set_xlim(left=np.min(x_coordinates), right=np.max(x_coordinates))
        # Hide the y-axis values
        ax.get_yaxis().set_ticks([])

        ax.set_xlabel('Time (UTC)')

        # Set the axis locators (doesn't affect the data values, only makes the plot look better)
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        fig.autofmt_xdate()
        # Rotate the data labels for a tight fit
        ax.xaxis.set_tick_params(rotation=70)

        # Cut out unnecessary empty space from figure
        plt.subplots_adjust(left=0.001, right=0.999, top=1.0, bottom=0.35)

        # Save the figure
        figure_path = folder_path / 'plots'
        file_id = "_".join((tr.stats.network, tr.stats.station, tr.stats.location,
                            tr.stats.channel, event_id))
        file_path = figure_path / (file_id + ".png")

        plt.savefig(file_path, dpi=300,  bbox_inches='tight', pad_inches=0)
        plt.cla()

    plt.close(fig)  # Clear memory


def generate_audio(event_id, st, origin, inv, sampling_rate, group_vel=4.5, surface_len=2000.0,
                   speed=400, damping=2e-8, folder_name="default_folder"):
    """
    Convert waveforms to audible sounds
    Args:
        event_id (str): Event ID -> Ideally should be time stamp YYYY-MM-DDTHH:MM:SS.000
        st (Obspy stream obj): Stream object which includes traces
        origin (Obspy origin obj): Origin object obtained for each event
        inv (Obspy inventory obj): Inventory information obtained using get_stations() func
        sampling_rate (float): Sampling rate in Hertz
        group_vel (float): Velocity at which waves move in km/s (Default=4.5; known value of
                          surface waves)
        surface_len (float): Surface window length in seconds which determines how long the
                            trimmed trace will be
        speed (int): Used to calculate audio sampling rate. Speed = 200 causes loss of low
                    frequencies, speed = 1600, causes loss of high frequencies. 400 is the default
                    "medium" speed rate
        damping (float): Amount by which to damp the sound. If damping is larger,
                        audio amplitude will be smaller. If lower, amplitude will be larger.
        folder_name (str): Name of the folder in which the data gets saved
    """
    # Make sure folders are created
    folder_path = create_folders(event_id,  folder_name=folder_name)

    # Apply band-pass filter to data stream
    st.filter('bandpass', freqmin=2.0, freqmax=8.0, zerophase=True)

    # Trim Seismograms
    for tr in st:
        coordinates = inv.get_coordinates(".".join((tr.stats.network, tr.stats.station,
                                                    tr.stats.location, tr.stats.channel)),
                                          datetime=origin.time)

        sta_lat, sta_lon = coordinates['latitude'], coordinates['longitude']
        # gps2dist returns distance (in meters) and forward, backward azimuths. Select the distance
        dist_origin_to_sta = gps2dist_azimuth(origin.latitude, origin.longitude, sta_lat,
                                              sta_lon)[0]

        # Dist / 1000 converts it to distance in km.
        # Then dividing it by group vel (km/s) converts it to seconds
        start = origin.time + (dist_origin_to_sta / 1000) / group_vel
        end = start + surface_len
        tr.trim(starttime=start, endtime=end, pad=True, nearest_sample=False, fill_value=0)
        tr.taper(max_percentage=None, type='hann', max_length=1.0, side='both')

        # Calculate the sampling rate
        new_sampling_rate = speed * sampling_rate
        duration = tr.stats.npts/new_sampling_rate

        # Scale the sound w.r.t arc tangent curve
        scaled_sound = (2**31)*np.arctan(tr.data/damping)*2/np.pi

        # Clean the event-id and generate the file-id
        event_id = clean_event_id(event_id)

        audio_path = folder_path / 'audio'
        file_id = "_".join((tr.stats.network, tr.stats.station, tr.stats.location,
                            tr.stats.channel, event_id, str(speed)))
        file_path = audio_path / (file_id + ".mp3")

        wavfile.write(file_path, rate=int(new_sampling_rate), data=np.int32(scaled_sound))

