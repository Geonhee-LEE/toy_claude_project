"""Setup configuration for mpc_controller_ros2 package."""

import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'mpc_controller_ros2'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Scripts
        ('share/' + package_name + '/scripts', glob('scripts/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='MPC Developer',
    maintainer_email='dev@example.com',
    description='ROS2 wrapper for MPC controller',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mpc_controller_node = mpc_controller_ros2.mpc_controller_node:main',
            'odom_to_tf = mpc_controller_ros2.odom_to_tf:main',
            'twist_to_twist_stamped = mpc_controller_ros2.twist_to_twist_stamped:main',
        ],
    },
)
