from obspy import read, read_inventory
from obspy import UTCDateTime
import matplotlib.pyplot as plt
import numpy as np
from obspy.clients.fdsn import Client
#
#
# st = read('http://examples.obspy.org/RJOB_061005_072159.ehz.new')
# st += read('http://examples.obspy.org/RJOB_061005_072159.ehe.new')
# st += read('http://examples.obspy.org/RJOB_061005_072159.ehn.new')


# st += read('http://examples.obspy.org/RJOB_061005_072159.ehz.new')
# st += read('http://examples.obspy.org/RJOB_061005_072159.ehz.new')

# st = read()
# threechannels = read('https://examples.obspy.org/COP.BHE.DK.2009.050')
# threechannels += read('https://examples.obspy.org/COP.BHN.DK.2009.050')
# threechannels += read('https://examples.obspy.org/COP.BHZ.DK.2009.050')
# print(threechannels.plot(size=(800, 600)))

# t1 = st[0]
# print(t1.stats)
# arr = t1.data
# inv = read_inventory()
# print(inv)
# start = t1.stats.starttime
# print(t1.plot(starttime=start+60, endtime=start+120))
# t1.remove_response(inventory=inv)
# print(t1.plot(starttime=))

# # print(list(arr))
#
# print("-----------------")
# # print(sorted(list(arr), reverse=True))
# print(arr.shape)
# print(t1.times("matplotlib"))

# st2 = read('../data/RawSeismograms/AK_A21K__BHZ_2020_03_25_T02_49_21_000.sac')
# st3 = read('../data/Seismograms/AK_A21K__BHZ_2020_03_25_T02_49_21_000.sac')
# print(st2)
# # print(st3.data)
# print(st3.stats)
# print(st3.plot())

# t1_filt = t1.copy()
# t1_filt.filter('lowpass', freq=5.0, corners=2, zerophase=True)
# t = np.arange(0, t1.stats.npts / t1.stats.sampling_rate, t1.stats.delta)
# plt.subplot(211)
# plt.plot(t, t1.data, 'k')
# plt.ylabel('Raw Data')
# plt.subplot(212)
# plt.plot(t, t1_filt.data, 'k')
# plt.ylabel('Lowpassed Data')
# plt.xlabel('Time [s]')
# plt.suptitle(t1.stats.starttime)
# plt.show()

# event = "2012-02-27"
# event_time = "T06:34:13.000"
# client = Client("IRIS")
# cat = client.get_events(eventid=609301)
# origin = cat[0].preferred_origin()
#
# print(origin)
# print("Time: ", origin.time)


a = 8
print(a / 2 / 2)