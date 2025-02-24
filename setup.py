from setuptools import find_packages, setup

package_name = 'common_road_sim'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'commonroad-vehicle-models', 'numpy', 'scipy', 'matplotlib', 'Pillow'],
    zip_safe=True,
    maintainer='Renukanandan Tumu',
    maintainer_email='nandan@nandantumu.com',
    description='This package contains a dynamics simulator for ROS2 based on CommonRoad vehicle models',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mb_simulator = common_road_sim.mb_simulator:main'
        ],
    },
)
