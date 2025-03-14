import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.transform import Rotation as R
import traceback
import yaml

#!/usr/bin/env python3


class FrictionSwitcher(Node):
    def __init__(self):
        super().__init__("friction_switcher")

        # Declare parameters

        self.map_metadata = (
            get_package_share_directory("common_road_sim")
            + "/resource/textures/map_metadata.yaml"
        )
        self.texture_images = []

        # Parse the map metadata
        try:
            with open(self.map_metadata, "r") as f:
                metadata = yaml.safe_load(f)
                # These are the (x,y) coordinates of the center slpitter for textures
                self.center_coordinate = metadata["center_coordinate"]
                self.friction_list = metadata["friction_list"]

        except Exception as e:
            self.get_logger().error(f"Error parsing map metadata: {e}")
            rclpy.shutdown()
            return

        # Initialize publishers and subscribers
        self.friction_pub = self.create_publisher(Float32, "/friction", 10)
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, "/ground_truth/pose", self.pose_callback, 10
        )

        self.get_logger().info("Friction Switcher node initialized.")

    def get_texture(self, pos):
        """
        Retrieve the appropriate texture based on the given position.

        -----------
        pos : tuple
            (x, y) position in world coordinates (meters).

        --------
        texture : int
            The texture index corresponding to the quadrant containing the given position.
            Selection is based on the position relative to the center coordinate.
        """
        x, y = pos
        # Get the texture corresponding to the gopro's position
        # Calculate texture index directly using boolean logic
        x_index = 2 if x >= self.center_coordinate[0] else 0
        y_index = 1 if y >= self.center_coordinate[1] else 0
        texture = x_index + y_index
        return texture

    def pose_callback(self, msg):
        """
        Callback function that processes the robot's pose and publishes the corresponding friction value.

        This method extracts the robot's position from the pose message, determines the ground texture
        at that position, looks up the corresponding friction coefficient, and publishes it.

        Args:
            msg: The pose message containing the robot's current position.
                Expected to have a pose.pose attribute with position information.

        Publishes:
            A Float32 message containing the friction coefficient to the configured friction topic.
        """
        # Get the transform from map to base_link
        transform = PoseStamped()
        transform.pose = msg.pose.pose

        # Robot's position in the map frame
        robot_x = transform.pose.position.x
        robot_y = transform.pose.position.y

        # Get the image
        texture = self.get_texture((robot_x, robot_y))

        # Return the friction value
        friction = self.friction_list[texture]
        friction_msg = Float32()
        friction_msg.data = friction
        self.friction_pub.publish(friction_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FrictionSwitcher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
