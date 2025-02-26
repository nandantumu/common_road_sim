import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory

#!/usr/bin/env python3


class VirtualGoProNode(Node):
    def __init__(self):
        super().__init__("virtual_go_pro")

        # Declare parameters
        # self.declare_parameter("image_path", "../resource/clark_park/ClarkPark.jpg")

        # self.declare_parameter("camera_info_url", "")  # Optional: URL for camera info

        # Get parameters
        self.image_path = (
            get_package_share_directory("common_road_sim")
            + "/resource/clark_park/ClarkPark.jpg"
        )

        # Load the image
        try:
            self.full_map = cv2.imread(self.image_path)
            if self.full_map is None:
                raise FileNotFoundError(f"Could not load image from {self.image_path}")
            self.map_height, self.map_width, _ = self.full_map.shape
            self.get_logger().info(
                f"Loaded map image with dimensions: {self.map_width}x{self.map_height}"
            )
        except FileNotFoundError as e:
            self.get_logger().error(str(e))
            rclpy.shutdown()
            return

        # Initialize publishers and subscribers
        self.image_pub = self.create_publisher(Image, "camera/image_raw", 10)
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, "/ground_truth/pose", self.pose_callback, 10
        )

        # Initialize TF buffer and listener
        # self.tf_buffer = tf2_ros.BufferClient(self)
        # self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self) #using buffer client instead

        # Initialize CV Bridge
        self.bridge = CvBridge()

        # Camera parameters (adjust these based on your desired camera view)
        self.camera_height = 1.5  # Height of the camera above the ground (in meters)
        self.camera_fov = 60.0  # Field of view in degrees
        self.image_width = 640
        self.image_height = 480
        self.pixels_per_meter = 140  # Adjust this based on your map's scale

        self.get_logger().info("Virtual GoPro node initialized.")

    def pose_callback(self, msg):
        """
        Callback function to process the robot's pose and generate a camera image.
        """
        try:
            # Get the transform from map to base_link
            transform = PoseStamped()
            transform.pose = msg.pose.pose

            # Robot's position in the map frame
            robot_x = transform.pose.position.x
            robot_y = transform.pose.position.y

            # Calculate the image center in map coordinates
            center_x = int(robot_x * self.pixels_per_meter)
            center_y = int(robot_y * self.pixels_per_meter)

            # Calculate the boundaries of the image in map coordinates
            half_width = int(self.image_width / 2)
            half_height = int(self.image_height / 2)
            x_min = center_x - half_width
            y_min = center_y - half_height
            x_max = center_x + half_width
            y_max = center_y + half_height

            # Extract the image from the map
            if (
                x_min >= 0
                and y_min >= 0
                and x_max < self.map_width
                and y_max < self.map_height
            ):
                image = self.full_map[y_min:y_max, x_min:x_max]
            else:
                # Handle cases where the camera view extends beyond the map boundaries
                image = np.zeros(
                    (self.image_height, self.image_width, 3), dtype=np.uint8
                )

                # Calculate valid boundaries
                x_min_valid = max(0, x_min)
                y_min_valid = max(0, y_min)
                x_max_valid = min(self.map_width, x_max)
                y_max_valid = min(self.map_height, y_max)

                # Extract the valid portion of the map
                valid_image = self.full_map[
                    y_min_valid:y_max_valid, x_min_valid:x_max_valid
                ]

                # Calculate the offsets for placing the valid image in the full image
                x_offset = x_min_valid - x_min
                y_offset = y_min_valid - y_min

                # Place the valid image in the full image
                image[
                    y_offset : y_offset + valid_image.shape[0],
                    x_offset : x_offset + valid_image.shape[1],
                ] = valid_image

            # Convert the image to a ROS Image message
            try:
                image_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
                image_msg.header.stamp = msg.header.stamp
                image_msg.header.frame_id = (
                    "camera_frame"  # You might want to create a camera frame
                )
                self.image_pub.publish(image_msg)
            except Exception as e:
                self.get_logger().error(f"Error converting image: {e}")

        except Exception as e:
            self.get_logger().warn(f"Some error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = VirtualGoProNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
