from setuptools import setup
import convert_weights

setup(
    name='convert_weights',
    version='1.0.0',
    install_requires=[
            'tqdm==4.64.1',
            'urllib3==1.26.13',
            'numpy==1.23.5',
            'torch==1.13.0+cu117',
            'h5py==3.7.0'
        ],
        entry_points={
    'console_scripts':
        ['convert_weights = convert_weights.convert_weights']
    }
)