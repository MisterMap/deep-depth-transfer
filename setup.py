#!/usr/bin/env python

import os
from distutils.core import setup

folder = os.path.dirname(os.path.realpath(__file__))
requirements_path = os.path.join(folder, 'requirements.txt')
install_requires = []
if os.path.isfile(requirements_path):
    with open(requirements_path) as f:
        install_requires = f.read().splitlines()

setup(name='DDT',
      version='0.1',
      description='Deep depth transfer',
      author='Deep project team',
      author_email='',
      package_dir={},
      packages=["DDT", "DDT.utils", "DDT.models", "DDT.data", "DDT.problems", "DDT.criterion", "DDT.data.kitty",
                "DDT.data.custom"],
      install_requires=install_requires
      )
