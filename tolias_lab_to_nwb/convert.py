import argparse
import os
import uuid

import numpy as np
from dateutil import parser
from pynwb import NWBFile, NWBHDF5IO
from pynwb.file import Subject
from pynwb.icephys import CurrentClampStimulusSeries, CurrentClampSeries, IZeroClampSeries
from ruamel import yaml
from scipy.io import loadmat

from .data_prep import data_preparation


# fpath = '/Users/bendichter/data/Berens/08 01 2019 sample 1.mat'


def gen_current_stim_template(times, rate):
    current_template = np.zeros((int(rate * times[-1]),))
    current_template[int(rate * times[0]):int(rate * times[1])] = 1

    return current_template


def make_nwb(current, voltage, rate, session_id, metadata, filepath):

    session_date = parser.parse(session_id[:10])

    nwbfile_args = dict(
        identifier=str(uuid.uuid4()),
        session_start_time=session_date,
        session_id=session_id
    )
    nwbfile_args.update(metadata['NWBFile'])
    nwbfile = NWBFile(**nwbfile_args)

    nwbfile.subject = Subject(**metadata['Subject'])

    device = nwbfile.create_device(**metadata['Icephys']['Device'])

    elec = nwbfile.create_ic_electrode(device=device,
                                       **metadata['Icephys']['Electrode'])

    current_template = gen_current_stim_template(times=(.1, .7, .9), rate=rate)

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

    with NWBHDF5IO(filepath, 'w') as io:
        io.write(nwbfile)

    #  test read
    with NWBHDF5IO(filepath, 'r') as io:
        io.read()


def main():
    argparser = argparse.ArgumentParser(
        description='convert .mat file to NWB',
        epilog="example usage:\n"
               "  python -m tolias_lab_to_nwb.convert '/path/to/08 01 2019 sample 1.mat'\n"
               "  python -m tolias_lab_to_nwb.convert '/path/to/08 01 2019 sample 1.mat' -m path/to/metafile.yml\n"
               "  python -m tolias_lab_to_nwb.convert '/path/to/08 01 2019 sample 1.mat' -m path/to/metafile.yml -o "
               "path/to/dest.nwb",
        formatter_class=argparse.RawTextHelpFormatter)
    argparser.add_argument("input_fpath", type=str, help="path of .mat file to convert")
    argparser.add_argument("-o", "--output_fpath", type=str, default=None,
                           help="path to save NWB file. If not provided, file will\n"
                                "output as input_fname.nwb in the same directory \n"
                                "as the input data.")
    argparser.add_argument("-m", "--metafile", type=str, default=None,
                           help="YAML file that contains metadata for experiment. \n"
                                "If not provided, will look for metafile.yml in the\n"
                                "same directory as the input data.")

    args = argparser.parse_args()

    fpath_base, fname = os.path.split(args.input_fpath)
    session_id = os.path.splitext(fname)[0]

    if not args.output_fpath:
        args.output_fpath = os.path.join(fpath_base, session_id + '.nwb')

    if not args.metafile:
        args.metafile = os.path.join(fpath_base, 'metafile.yml')

    with open(args.metafile) as f:
        metadata = yaml.safe_load(f)

    data = loadmat(args.input_fpath)
    time, current, voltage, curr_index_0 = data_preparation(data)
    make_nwb(current, voltage, 25e3, session_id, metadata, args.output_fpath)


if __name__ == "__main__":
    main()
