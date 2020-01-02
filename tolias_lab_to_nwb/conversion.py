import os
import uuid

import numpy as np
from data_prep import data_preparation
from dateutil import parser
from pynwb import NWBFile, NWBHDF5IO
from pynwb.icephys import CurrentClampStimulusSeries, CurrentClampSeries, IZeroClampSeries
from scipy.io import loadmat

fpath='/Users/bendichter/data/Berens/08 01 2019 sample 1.mat'

fpath_base, fname = os.path.split(fpath)
session_id = os.path.splitext(fname)[0]
data = loadmat(fpath)

session_date = parser.parse(os.path.split(fpath)[1][:10])

time, current, voltage, curr_index_0 = data_preparation(data)

nwbfile = NWBFile('Current clamp square',
                  str(uuid.uuid4()),
                  session_date,
                  experimenter='Federico Scala',
                  lab='Tolias',
                  institution='Baylor College of Medicine',
                  experiment_description='Current clamp square',
                  session_id=session_id)

device = nwbfile.create_device(name='Heka ITC-1600')

elec = nwbfile.create_ic_electrode(name="elec0",
                                   description='an intracellular electrode',
                                   device=device)

times = [.1, .7, .9]
rate = 25e3
current_template = np.zeros((int(rate * times[-1]),))
current_template[int(rate * times[0]):int(rate * times[1])] = 1

for i, (ivoltage, icurrent) in enumerate(zip(voltage.T, current)):

    if icurrent == 0:
        nwbfile.add_acquisition(IZeroClampSeries(
            name="CurrentClampSeries{:03d}".format(i),
            data=ivoltage,
            electrode=elec,
            rate=rate,
            gain=1.,
            sweep_number=i))
    else:
        nwbfile.add_stimulus(CurrentClampStimulusSeries(
            name="CurrentClampStimulusSeries{:03d}".format(i),
            data=current_template * icurrent,
            starting_time=np.nan,
            rate=rate,
            electrode=elec,
            gain=1.,
            sweep_number=i))

        nwbfile.add_acquisition(CurrentClampSeries(
            name="CurrentClampSeries{:03d}".format(i),
            data=ivoltage,
            electrode=elec,
            rate=rate,
            gain=1.,
            sweep_number=i))

out_fpath = session_id + '.nwb'
with NWBHDF5IO(out_fpath, 'w') as io:
    io.write(nwbfile)

#  test read
with NWBHDF5IO(out_fpath, 'r') as io:
    io.read()
