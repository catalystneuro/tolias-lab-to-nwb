# tolias-lab-to-nwb
Code for converting Tolias Lab data to NWB

## Installation
```shell script
pip install https://github.com/ben-dichter-consulting/tolias-lab-to-nwb.git
```

## Usage
convert data using a bash script:
```shell script
usage: convert.py [-h] [-o OUTPUT_FPATH] [-m METAFILE] input_fpath

convert .mat file to NWB

positional arguments:
  input_fpath           path of .mat file to convert

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_FPATH, --output_fpath OUTPUT_FPATH
                        path to save NWB file. If not provided, file will
                        output as input_fname.nwb in the same directory 
                        as the input data.
  -m METAFILE, --metafile METAFILE
                        YAML file that contains metadata for experiment. 
                        If not provided, will look for metafile.yml in the
                        same directory as the input data.

example usage:
  python -m tolias_lab_to_nwb.convert '/path/to/08 01 2019 sample 1.mat'
  python -m tolias_lab_to_nwb.convert '/path/to/08 01 2019 sample 1.mat' -m path/to/metafile.yml
  python -m tolias_lab_to_nwb.convert '/path/to/08 01 2019 sample 1.mat' -m path/to/metafile.yml -o path/to/dest.nwb
```

