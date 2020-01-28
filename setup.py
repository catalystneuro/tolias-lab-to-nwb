from setuptools import setup, find_packages

# Get the long description from the README file
with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='tolias_lab_to_nwb',
    version='0.3.0',
    description='tool to convert Tolias Lab matlab intracellular electrophysiology data into NWB:N format',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Ben Dichter',
    author_email='ben.dichter@gmail.com',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['pandas',
                      'numpy',
                      'scipy',
                      'ruamel.yaml',
                      'pynwb',
                      'nwbn_conversion_tools==0.2.0'
                      ]
)
