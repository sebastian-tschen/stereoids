import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
VERSION = open(os.path.join(here, 'VERSION.txt')).read()

REQUIREMENTS = [
    'opencv-contrib-python',
    'numpy',
]

setup(
    name='stereoids',
    version=VERSION,
    packages=find_packages(),
    url='https://github.com/sebastian-tschen/stereoids',
    license='MIT',
    author='Sebastian Behrens',
    author_email='sebastian.tschen.behrens@gmail.com',
    description='a simple library for 3d positioning markers via stereo vision',
    install_requires=REQUIREMENTS,
)
