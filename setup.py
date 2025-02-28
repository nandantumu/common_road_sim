from setuptools import find_packages, setup
from glob import glob
import os

package_name = "common_road_sim"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (
            os.path.join("share", package_name, "resource/clark_park"),
            glob("resource/clark_park/*"),
        ),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=[
        "setuptools",
        "commonroad-vehicle-models",
        "numpy",
        "scipy",
        "matplotlib",
        "Pillow",
    ],
    zip_safe=True,
    maintainer="Renukanandan Tumu",
    maintainer_email="nandan@nandantumu.com",
    description="This package contains a dynamics simulator for ROS2 based on CommonRoad vehicle models",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mb_simulator = common_road_sim.mb_simulator:main",
            "virtual_go_pro = common_road_sim.virtual_go_pro:main",
            "controller_simulator = common_road_sim.controller_simulator:main",
        ],
    },
)
