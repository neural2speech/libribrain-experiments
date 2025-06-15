from setuptools import setup, find_packages

setup(
    name='libribrain_experiments',
    version='0.0.1',
    author='miro-code',
    description='Experiments for LibriBrain',
    packages=find_packages(),
    install_requires=[
        'pnpl',
        'torch',
        'torchvision',
        'torchmetrics',
        'torchaudio',
        'lightning',
        'wandb',
        'numpy',
        'scikit-learn',
        'mne',
        'matplotlib',
        'h5py',
        'mne-bids',
        'transformers',
    ],
)
