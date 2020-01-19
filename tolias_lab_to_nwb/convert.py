import argparse
import os

import numpy as np
from dateutil import parser
from pynwb.icephys import CurrentClampStimulusSeries, CurrentClampSeries, IZeroClampSeries
from ruamel import yaml
from scipy.io import loadmat
from nwbn_conversion_tools import NWBConverter
from hdmf.backends.hdf5 import H5DataIO

from .data_prep import data_preparation


# fpath = '/Users/bendichter/data/Berens/08 01 2019 sample 1.mat'


def gen_current_stim_template(times, rate):
    current_template = np.zeros((int(rate * times[-1]),))
    current_template[int(rate * times[0]):int(rate * times[1])] = 1

    return current_template


class ToliasNWBConverter(NWBConverter):
    def add_icephys_data(self, current, voltage, rate):

        current_template = gen_current_stim_template(times=(.1, .7, .9), rate=rate)

        elec = list(self.ic_elecs.values())[0]

        for i, (ivoltage, icurrent) in enumerate(zip(voltage.T, current)):

            ccs_args = dict(
                name="CurrentClampSeries{:03d}".format(i),
                data=H5DataIO(ivoltage, compression=True),
                electrode=elec,
                rate=rate,
                gain=1.,
                starting_time=np.nan,
                sweep_number=i)
            if icurrent == 0:
                self.nwbfile.add_acquisition(IZeroClampSeries(**ccs_args))
            else:
                self.nwbfile.add_acquisition(CurrentClampSeries(**ccs_args))
                self.nwbfile.add_stimulus(CurrentClampStimulusSeries(
                    name="CurrentClampStimulusSeries{:03d}".format(i),
                    data=H5DataIO(current_template * icurrent, compression=True),
                    starting_time=np.nan,
                    rate=rate,
                    electrode=elec,
                    gain=1.,
                    sweep_number=i))


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

    metadata['NWBFile']['session_start_time'] = parser.parse(session_id[:10])
    metadata['NWBFile']['session_id'] = session_id

    tolias_converter = ToliasNWBConverter(metadata)

    data = loadmat(args.input_fpath)
    time, current, voltage, curr_index_0 = data_preparation(data)

    tolias_converter.add_icephys_data(current, voltage, rate=25e3)

    tolias_converter.save(args.output_fpath)


if __name__ == "__main__":
    main()
