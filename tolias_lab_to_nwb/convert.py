import argparse

import numpy as np
from pynwb.icephys import CurrentClampStimulusSeries, CurrentClampSeries, IZeroClampSeries
from nwbn_conversion_tools import NWBConverter
from hdmf.backends.hdf5 import H5DataIO
import pandas as pd

import os
import uuid

from dateutil import parser
from ruamel import yaml
from scipy.io import loadmat
from glob import glob
from tqdm import tqdm

from .data_prep import data_preparation

from ndx_icephys_meta import ICEphysFile


def gen_current_stim_template(times, rate):
    current_template = np.zeros((int(rate * times[-1]),))
    current_template[int(rate * times[0]):int(rate * times[1])] = 1

    return current_template


class ToliasNWBConverter(NWBConverter):
    def create_NWBFile(self, **NWBFile_metadata):
        """

        Parameters
        ----------
        NWBFile_metadata: args to go into NWBFile

        Returns
        -------

        """
        nwbfile_args = dict(identifier=str(uuid.uuid4()), )
        nwbfile_args.update(**NWBFile_metadata)
        return ICEphysFile(**nwbfile_args)

    def add_icephys_data(self, current, voltage, rate):

        current_template = gen_current_stim_template(times=(.1, .7, .9), rate=rate)

        elec = list(self.ic_elecs.values())[0]
        sweep_indexes = []
        for i, (ivoltage, icurrent) in enumerate(zip(voltage.T, current)):

            ccs_args = dict(
                name="CurrentClampSeries{:03d}".format(i),
                data=H5DataIO(ivoltage, compression=True),
                electrode=elec,
                rate=rate,
                gain=1.,
                starting_time=np.nan)
            if icurrent == 0:
                response = IZeroClampSeries(**ccs_args)
                self.nwbfile.add_acquisition(response)
                ir_index = self.nwbfile.add_intracellular_recording(electrode=elec, response=response)
            else:
                response = CurrentClampSeries(**ccs_args)
                self.nwbfile.add_acquisition(response)

                stimulus = CurrentClampStimulusSeries(
                    name="CurrentClampStimulusSeries{:03d}".format(i),
                    data=H5DataIO(current_template * icurrent, compression=True),
                    starting_time=np.nan,
                    rate=rate,
                    electrode=elec,
                    gain=1.)
                self.nwbfile.add_stimulus(stimulus)
                ir_index = self.nwbfile.add_intracellular_recording(electrode=elec, stimulus=stimulus,
                                                                    response=response)
            sweep_indexes.append(self.nwbfile.add_ic_sweep(recordings=[ir_index, ]))
        self.nwbfile.add_ic_sweep_sequence(sweeps=sweep_indexes)


def fetch_metadata(lookup_tag, csv, metadata):
    """Reads metadata from mini-atlas-meta-data.csv and inserts it in the metadata object

    Parameters
    ----------
    lookup_tag: str
        e.g. '20190403_sample_5'
    csv: path to mini-atlas-meta-data.csv
    metadata: dict
        metadata object

    Returns
    -------

    """
    df = pd.read_csv(csv, sep='\t')

    metadata_from_csv = df[df['Cell'] == lookup_tag]

    if 'Subject' not in metadata:
        metadata['Subject'] = dict()
    metadata['Subject']['subject_id'] = metadata_from_csv['Mouse'].iloc[0]
    metadata['Subject']['date_of_birth'] = parser.parse(metadata_from_csv['Mouse date of birth'].iloc[0])

    user = metadata_from_csv['User'].iloc[0]
    user_map = {
        'Fede': 'Federico Scala',
        'Matteo': 'Matteo Bernabucci'
    }
    metadata['NWBFile']['experimenter'] = user_map[user]

    return metadata


def convert_all(data_dir='/Volumes/easystore5T/data/Tolias/ephys',
                metafile_fpath='/Users/bendichter/dev/tolias-lab-to-nwb/metafile.yml',
                out_dir='/Volumes/easystore5T/data/Tolias/nwb',
                meta_csv_file='/Volumes/easystore5T/data/Tolias/ephys/mini-atlas-meta-data.csv'):

    fpaths = glob(os.path.join(data_dir, '*/*/*/*.mat'))

    for input_fpath in tqdm(fpaths):
        fpath_base, fname = os.path.split(input_fpath)
        old_session_id = os.path.splitext(fname)[0]
        session_start_time = parser.parse(old_session_id[:10])
        samp_str = '_'.join(old_session_id.split(' ')[-2:])
        lookup_tag = session_start_time.strftime("%Y%m%d") + '_' + samp_str
        output_fpath = os.path.join(out_dir, lookup_tag + '.nwb')

        # handle metadata
        with open(metafile_fpath) as f:
            metadata = yaml.safe_load(f)

        # load in session-specific data
        # session_start_time = session_start_time.replace(tzinfo=gettz('US/Central'))
        metadata['NWBFile']['session_start_time'] = session_start_time
        metadata['NWBFile']['session_id'] = lookup_tag

        metadata = fetch_metadata(lookup_tag, meta_csv_file, metadata)

        tolias_converter = ToliasNWBConverter(metadata)

        data = loadmat(input_fpath)
        time, current, voltage, curr_index_0 = data_preparation(data)

        tolias_converter.add_icephys_data(current, voltage, rate=25e3)

        tolias_converter.save(output_fpath)


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
