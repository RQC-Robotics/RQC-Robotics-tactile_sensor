from setuptools import setup, find_packages
from os.path import join, dirname

setup(
    name='tactil_sensor_decoder',
    version='1.0',
    author='korbash and Artem',
    author_email='korbash179@gmail.com',
    py_modules=['sensor_lib']
    # long_description=open(join(dirname(__file__), 'README.txt')).read(),
)