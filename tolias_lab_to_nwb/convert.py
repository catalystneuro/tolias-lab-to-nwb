import argparse
import os
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
from dateutil import parser
from hdmf.backends.hdf5 import H5DataIO
from joblib import Parallel, delayed
from ndx_dandi_icephys import DandiIcephysMetadata, CreSubject
from pynwb.icephys import CurrentClampStimulusSeries, CurrentClampSeries, IZeroClampSeries
import uuid
from pynwb import NWBFile, NWBHDF5IO
from ruamel import yaml
from scipy.io import loadmat
from tqdm import tqdm

from .data_prep import data_preparation
from .parallel import tqdm_joblib


def gen_current_stim_template(times, rate):
    current_template = np.zeros((int(rate * times[-1]),))
    current_template[int(rate * times[0]):int(rate * times[1])] = 1

    return current_template


class ToliasNWBConverter:

    def __init__(self, metadata):
        kwargs = metadata['NWBFile']
        kwargs['identifier'] = metadata['NWBFile'].get('identifier', str(uuid.uuid4()))

        self.nwbfile = NWBFile(**metadata['NWBFile'])

    def add_meta_data(self, metadata):
        self.nwbfile.add_lab_meta_data(DandiIcephysMetadata(**metadata))

    def create_subject(self, metadata_subject):
        print(metadata_subject)
        self.nwbfile.subject = CreSubject(**metadata_subject)

    def add_icephys_data(self, current, voltage, rate):

        current_template = gen_current_stim_template(times=(.1, .7, .9), rate=rate)

        device = self.nwbfile.create_device('device')

        elec = self.nwbfile.create_icephys_electrode(
            name="elec0",
            description='a mock intracellular electrode',
            device=device
        )

        for i, (ivoltage, icurrent) in enumerate(zip(voltage.T, current)):

            ccs_args = dict(
                name="CurrentClampSeries{:03d}".format(i),
                data=H5DataIO(ivoltage, compression=True),
                electrode=elec,
                rate=rate,
                gain=1.,
                starting_time=np.nan,
                sweep_number=np.uint32(i)
            )
            if icurrent == 0:
                self.nwbfile.add_acquisition(IZeroClampSeries(**ccs_args))
            else:
                self.nwbfile.add_acquisition(CurrentClampSeries(**ccs_args))
                self.nwbfile.add_stimulus(
                    CurrentClampStimulusSeries(
                        name="CurrentClampStimulusSeries{:03d}".format(i),
                        data=H5DataIO(current_template * icurrent, compression=True),
                        starting_time=np.nan,
                        rate=rate,
                        electrode=elec,
                        gain=1.,
                        sweep_number=np.uint32(i)
                    )
                )

    def save(self, output_path):
        with NWBHDF5IO(output_path, 'w') as io:
            io.write(self.nwbfile)


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
        metadata['Subject'] = dict(
            species='Mus musculus',
            subject_id=metadata_from_csv['Mouse'].iloc[0],
            date_of_birth=parser.parse(metadata_from_csv['Mouse date of birth'].iloc[0]),
            age='P{}D'.format(str(metadata_from_csv['Mouse age'].iloc[0])),
            sex=metadata_from_csv['Mouse gender'].iloc[0],
            genotype=metadata_from_csv['Mouse genotype'].iloc[0],
            cre=metadata_from_csv['Cre'].iloc[0],
        )

    user = metadata_from_csv['User'].iloc[0]
    user_map = {
        'Fede': 'Federico Scala',
        'Matteo': 'Matteo Bernabucci',
        'Fede, Ray': ['Federico Scala', 'Jesus Ramon Castro']
    }
    metadata['NWBFile']['experimenter'] = user_map[user]

    metadata['lab_meta_data'] = {'cell_id': lookup_tag}
    metadata_dtype = {
        'Length (bp)': int,
        'Yield (pg/µl)': int,
        'Traced': str,
        'Exclusion reasons': str,
        'Soma depth (µm)': float
    }
    for nwb_key, csv_key in (('slice_id', 'Slice'),
                             ('targeted_layer', 'Targeted layer'),
                             ('inferred_layer', 'Inferred layer'),
                             ('exon_reads', 'Exon reads'),
                             ('intron_reads', 'Intron reads'),
                             ('intergenic_reads', 'Intergenic reads'),
                             ('sequencing_batch', 'Sequencing batch'),
                             ('number_of_genes_detected', 'Number of genes detected'),
                             ('RNA_family', 'RNA family'),
                             ('RNA_type', 'RNA type'),
                             ('RNA_type_confidence', 'RNA type confidence'),
                             ('RNA_type_top3', 'RNA type top-3'),
                             ('ALM_VISp_top3', 'ALM/VISp top-3'),
                             ('length', 'Length (bp)'),
                             ('yield', 'Yield (pg/µl)'),
                             ('hold_time (min)', 'Hold Time (min'),
                             ('soma_depth_4x', 'soma_depth (4x)'),
                             ('soma_depth_um', 'Soma depth (µm)'),
                             ('cortical_thickness_4x', 'Soma depth (4x)'),
                             ('cortical_thickness_um', 'Cortical thickness (4x)'),
                             ('traced', 'Traced'),
                             ('exclusion_reason', 'Exclusion reasons'),
                             ('recording_temperature', 'Recording Temperature (˚C)')):
        if csv_key in metadata_from_csv and metadata_from_csv[csv_key].iloc[0]:
            if csv_key in metadata_dtype:
                try:
                    input_val = metadata_from_csv[csv_key].iloc[0]
                    if np.isnan(input_val) and metadata_dtype[csv_key] is str:
                        val = None
                    else:
                        val = metadata_dtype[csv_key](input_val)
                except:
                    val = None
            else:
                val = metadata_from_csv[csv_key].iloc[0]
                # catch case where missing strings are NaNs
                if isinstance(val, np.float) and np.isnan(val):
                    val = None
            metadata['lab_meta_data'].update({nwb_key: val})

    if 'Pipette Resistance (MΩ)' in metadata_from_csv:
        metadata['Icephys'].update(
            Electrode=dict(
                resistance=
                "Pipette Resistance (MΩ):{}\n"
                "Access Resistance (MΩ): {}\n"
                "Seal Resistance (MΩ): {}".format(
                    *metadata_from_csv[['Pipette Resistance (MΩ)',
                                        'Access Resistance (MΩ)',
                                        'Seal Resistance (MΩ)']].iloc[0]
                )
            )
        )

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
    tolias_converter.create_subject(metadata['Subject'])

    data = loadmat(input_fpath)
    time, current, voltage, curr_index_0 = data_preparation(data)

    tolias_converter.add_icephys_data(current, voltage, rate=25e3)
    tolias_converter.add_meta_data(metadata['lab_meta_data'])

    tolias_converter.save(output_fpath)


def convert_all(data_dir='/Volumes/easystore5T/data/Tolias/ephys',
                metafile_fpath='/Users/bendichter/dev/tolias-lab-to-nwb/metafile.yml',
                out_dir='/Volumes/easystore5T/data/Tolias/nwb',
                meta_csv_file='/Volumes/easystore5T/data/Tolias/ephys/m1_patchseq_meta_data.csv',
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
    tolias_converter.create_subject(metadata['Subject'])

    data = loadmat(args.input_fpath)
    time, current, voltage, curr_index_0 = data_preparation(data)

    tolias_converter.add_icephys_data(current, voltage, rate=25e3)

    tolias_converter.save(args.output_fpath)


if __name__ == "__main__":
    main()
