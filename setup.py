from io import open
from setuptools import setup, find_packages
from pathlib import Path


"""
:authors: Elisha Flacon
:license: BSD 3-Clause License, see LICENSE file
:copyright: (c) 2023 Elisha Flacon
"""


VERSION = '0.0.1.0'
NAME = 'signess'

HERE = Path(__file__).parent.resolve()

AUTHOR = 'Elisha Flacon'
DESCRIPTION = 'Signess'
LONG_DESCRIPTION = Path(HERE, 'README.md').read_text(encoding='utf-8')

URL = 'https://github.com/ElishaFlacon/signess'
DOWNLOAD_URL = 'https://github.com/ElishaFlacon/signess/archive/main.zip'

REQUIRES_PYTHON = '>=3.8'
LICENSE = 'BSD 3-Clause'


setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,

    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',

    url=URL,
    download_url=DOWNLOAD_URL,

    python_requires=REQUIRES_PYTHON,

    license=LICENSE,

    packages=find_packages(),
    include_package_data=True,
    install_requires=['reepsy', 'inskrib', 'fedot[extra]'],

    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ]
)
