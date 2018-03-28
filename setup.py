import os
from setuptools import setup
from configparser import ConfigParser

# Read in the requirements.txt file
with open('./requirements.txt') as f:
    requirements = f.read().splitlines()

config = ConfigParser()
config.read(os.path.abspath(os.path.join(os.path.dirname(__file__), './python-assembly-line.cfg')))

setup(
    name='python-assembly-line',
    version=config['DEFAULT']['version'],
    install_requires=requirements,
    author='Luis Visintini',
    author_email='lvisintini@gmail.com',
    packages=['assembly_line', ],
    url='https://github.com/lvisintini/python-assembly-line',
    license='The MIT License (MIT)',
    description='A python toolkit to define a assembly line of task and run "things" through it.',
)