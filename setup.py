#!/usr/bin/env python

import os
from distutils.core import setup

folder = os.path.dirname(os.path.realpath(__file__))
requirements_path = os.path.join(folder, 'requirements.txt')
install_requires = []
if os.path.isfile(requirements_path):
    with open(requirements_path) as f:
        install_requires = f.read().splitlines()

setup(name='deep_depth_transfer',
      version='0.1',
      description='Deep depth transfer',
      author='Deep project team',
      author_email='',
      package_dir={},
      packages=["deep_depth_transfer", "deep_depth_transfer.utils", "deep_depth_transfer.models",
                "deep_depth_transfer.data", "deep_depth_transfer.criterion", "deep_depth_transfer.data.kitti",
                "deep_depth_transfer.data.skoltech", "deep_depth_transfer.data.tum_rgbd", "deep_depth_transfer.data.custom",
                "deep_depth_transfer.data.kitti_eigen",],
      install_requires=install_requires
      )
