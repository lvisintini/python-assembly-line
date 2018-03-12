from setuptools import setup
from configparser import ConfigParser

# Read in the requirements.txt file
with open('./requirements.txt') as f:
    requirements = f.read().splitlines()

config = ConfigParser()
config.read('./python-production-line.cfg')

setup(
    name='python-production-line',
    version=config['default']['version'],
    install_requires=requirements,
    author='Luis Visintini',
    author_email='lvisintini@gmail.com',
    packages=['production_line', ],
    url='https://github.com/lvisintini/python-production-line',
    license='The MIT License (MIT)',
    description='A python toolkit to define a production line of task and run "things" through it.',
)