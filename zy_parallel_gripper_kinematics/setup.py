from distutils.core import setup
from setuptools import find_packages
from setuptools.command.install import install
import os

setup(
    name='zy_parallel_gripper_layer',
    version='1.0.0',
    description='Hand kinematics layer for ZY Parallel Gripper',
    author='Wei Wei',
    author_email='weiwei72607260@gmail.com',
    url='https://github.com/v-wewei/parallel_grasp_annotation',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'trimesh',
        'roma',
    ]
)

