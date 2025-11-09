from glob import glob
import os

from setuptools import setup


package_name = 'realtime_trees_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
        (os.path.join('share', package_name, 'config', 'procman'), glob(os.path.join('config', 'procman', '*.pmd'))),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'rviz'), glob(os.path.join('rviz', '*.rviz')))
    ],
    entry_points={
        'console_scripts': [
             'realtime_trees_node = realtime_trees_ros.realtime_trees_node:main',
             'payload_accumulator_node = realtime_trees_ros.payload_accumulator_node:main',
        ],
    },
    install_requires=['setuptools']
)
