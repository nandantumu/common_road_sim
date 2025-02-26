#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Nodes to launch
    video = Node(
        package="common_road_sim",
        executable="virtual_go_pro",
        name="virtual_go_pro",
        output="screen",
    )

    mb_simulator = Node(
        package="common_road_sim",
        executable="mb_simulator",
        name="mb_simulator",
        output="screen",
    )

    return LaunchDescription([video, mb_simulator])
