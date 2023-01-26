from setuptools import setup, find_packages
import weights_converter
setup(
    name='weights_converter',
    version=weights_converter.__version__,
    packages=find_packages(),
    install_requires=[
            'tqdm==4.64.1',
            'urllib3==1.26.13',
            'numpy==1.23.5',
            'torch==1.13.0',
            'h5py==3.7.0'
        ],
        entry_points={
    'console_scripts':
        ['weights_converter = weights_converter.__main__:main']
    }
)