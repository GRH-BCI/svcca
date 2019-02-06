#!/usr/bin/env python

from distutils.core import setup
import setuptools
import os

root_dir = os.path.abspath(os.path.dirname(__file__))

with open(f'{root_dir}/README.md') as f:
    readme = f.read()

with open(f'{root_dir}/requirements.txt') as f:
    requirements = f.read().split()

packages = setuptools.find_packages('.', include='svcca.*')

setup(name='svcca',
      version='0.0.1',
      description='SVCCA on Numpy, Cupy, and PyTorch',
      long_description=readme,
      author='Rasmus Diederichsen',
      author_email='rasmus@peltarion.com',
      url='https://github.com/themightyoarfish/svcca-gpu',
      classifiers=['Intended Audience :: Science/Research',
                   'Topic :: Scientific/Engineering :: Visualization',
                   'Programming Language :: Python :: 3.6',
                   'License :: OSI Approved :: Apache License',
                   'Intended Audience :: Developers',
                   ],
      keywords='deep-learning pytorch cupy numpy svcca neural-networks machine-learning'.split(),
      install_requires=requirements,
      packages=packages,
      zip_safe=False,   # don't install egg, but source
      )
