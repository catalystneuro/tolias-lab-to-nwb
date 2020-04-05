import argparse
import os
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
from dateutil import parser
from hdmf.backends.hdf5 import H5DataIO
from joblib import Parallel, delayed
from ndx_dandi_icephys import DandiIcephysMetadata
from nwbn_conversion_tools import NWBConverter
from pynwb.icephys import CurrentClampStimulusSeries, CurrentClampSeries, IZeroClampSeries
from ruamel import yaml
from scipy.io import loadmat
from tqdm import tqdm

from .data_prep import data_preparation
from .parallel import tqdm_joblib


def gen_current_stim_template(times, rate):
    current_template = np.zeros((int(rate * times[-1]),))
    current_template[int(rate * times[0]):int(rate * times[1])] = 1

    return current_template


class ToliasNWBConverter(NWBConverter):

    def add_meta_data(self, metadata):
        self.nwbfile.add_lab_meta_data(DandiIcephysMetadata(**metadata))

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
    if not any(df['Cell'] == lookup_tag):
        print('{} not found in {}'.format(lookup_tag, csv))
        return None
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
    metadata['lab_meta_data'] = {'cell_id': lookup_tag,
                                 'slice_id': metadata_from_csv['Slice'].iloc[0]}
    print(metadata['lab_meta_data'])

    return metadata


def convert_file(input_fpath, output_fpath, metafile_fpath, meta_csv_file, overwrite=False):
    if not overwrite and os.path.isfile(output_fpath):
        return
    fpath_base, fname = os.path.split(input_fpath)
    old_session_id = os.path.splitext(fname)[0]
    try:
        session_start_time = datetime.strptime(old_session_id[:10], '%m %d %Y')  # 04 18 2018
    except ValueError:
        session_start_time = datetime.strptime(old_session_id[:8], '%m%d%Y')  # 04182018
    samp_str = '_'.join(old_session_id.split(' ')[-2:])
    lookup_tag = session_start_time.strftime("%Y%m%d") + '_' + samp_str

    # handle metadata
    with open(metafile_fpath) as f:
        metadata = yaml.safe_load(f)

    # load in session-specific data
    # session_start_time = session_start_time.replace(tzinfo=gettz('US/Central'))
    metadata['NWBFile']['session_start_time'] = session_start_time
    metadata['NWBFile']['session_id'] = lookup_tag

    metadata = fetch_metadata(lookup_tag, meta_csv_file, metadata)
    if metadata is None:
        print('no metadata found for {}'.format(lookup_tag))
        return

    tolias_converter = ToliasNWBConverter(metadata)

    data = loadmat(input_fpath)
    time, current, voltage, curr_index_0 = data_preparation(data)

    tolias_converter.add_icephys_data(current, voltage, rate=25e3)
    tolias_converter.add_meta_data(metadata['lab_meta_data'])

    tolias_converter.save(output_fpath)


def convert_all(data_dir='/Volumes/easystore5T/data/Tolias/ephys',
                metafile_fpath='/Users/bendichter/dev/tolias-lab-to-nwb/metafile.yml',
                out_dir='/Volumes/easystore5T/data/Tolias/nwb',
                meta_csv_file='/Volumes/easystore5T/data/Tolias/ephys/mini-atlas-meta-data.csv',
                overwrite=False, n_jobs=1):

    fpaths = list(glob(os.path.join(data_dir, '*/*/*/*.mat')))

    with tqdm_joblib(tqdm(desc="Converting Tolias data", total=len(fpaths))) as progress_bar:
        Parallel(n_jobs=n_jobs)(delayed(gather_and_convert)
                                (input_fpath, out_dir, metafile_fpath, meta_csv_file, overwrite)
                                for input_fpath in fpaths)


def gather_and_convert(input_fpath, out_dir, metafile_fpath, meta_csv_file, overwrite=False):
    fpath_base, fname = os.path.split(input_fpath)
    old_session_id = os.path.splitext(fname)[0]
    try:
        session_start_time = datetime.strptime(old_session_id[:10], '%m %d %Y')  # 04 18 2018
    except ValueError:
        session_start_time = datetime.strptime(old_session_id[:8], '%m%d%Y')  # 04182018
    samp_str = '_'.join(old_session_id.split(' ')[-2:])
    lookup_tag = session_start_time.strftime("%Y%m%d") + '_' + samp_str
    output_fpath = os.path.join(out_dir, lookup_tag + '.nwb')
    if overwrite or not os.path.isfile(output_fpath):
        convert_file(input_fpath, output_fpath, metafile_fpath, meta_csv_file, overwrite)


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
