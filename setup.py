from setuptools import setup


REQUIREMENTS = [
    'opencv-contrib-python',
    'numpy',
]


setup(
    name='stereoids',
    version='0.0.2',
    packages=['detector', 'corrector', 'stereoids'],
    url='https://github.com/sebastian-tschen/stereoids',
    license='MIT',
    author='Sebastian Behrens',
    author_email='sebastian.tschen.behrens@gmail.com',
    description='a simple library for 3d positioning markers via stereo vision',
    install_requires=REQUIREMENTS,
)
