import rclpy
from rclpy.node import Node
import rclpy.time
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.transform import Rotation as R
import traceback
import yaml

#!/usr/bin/env python3


class VirtualGoProNode(Node):
    def __init__(self):
        super().__init__("virtual_go_pro")

        # Declare parameters
        # self.declare_parameter("image_path", "../resource/clark_park/ClarkPark.jpg")

        # self.declare_parameter("camera_info_url", "")  # Optional: URL for camera info
        self.declare_parameter("frequency", 60.0)  # Frequency of the camera publishing
        self.frequency = self.get_parameter("frequency").value

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
                self.texture_list = metadata["texture_list"]
                for texture in self.texture_list:
                    texture_path = (
                        get_package_share_directory("common_road_sim")
                        + "/resource/textures/"
                        + texture
                    )
                    loaded_texture = cv2.imread(texture_path)
                    if loaded_texture is None:
                        self.get_logger().error(
                            f"Failed to load texture image at {texture_path}"
                        )
                    else:
                        self.texture_images.append(loaded_texture)

        except Exception as e:
            self.get_logger().error(f"Error parsing map metadata: {e}")
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

        self.image_width = 640
        self.image_height = 480

        self.time = self.get_clock().now()

        self.get_logger().info("Virtual GoPro node initialized.")

    def simulate_gopro_view(self, gopro_pos, angle=0):
        """
        Extract the gopro's view from the large image.

        Parameters:
        gopro_pos     : (x, y) gopro's center position in world coordinates (meters).
        angle         : Rotation angle in radians (yaw) for the camera.

        Returns:
        captured_view: The sub-image corresponding to the gopro's current view.
        """
        gopro_x, gopro_y = gopro_pos
        # Get the texture corresponding to the gopro's position
        # Calculate texture index directly using boolean logic
        x_index = 2 if gopro_x >= self.center_coordinate[0] else 0
        y_index = 1 if gopro_y >= self.center_coordinate[1] else 0
        texture = self.texture_images[x_index + y_index]

        # Get the gopro's view
        # Rotate the texture to match the gopro's orientation
        rows, cols, _ = texture.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), np.degrees(angle), 1)
        rotated_texture = cv2.warpAffine(
            texture, M, (cols, rows), borderMode=cv2.BORDER_WRAP
        )
        captured_view = cv2.resize(
            rotated_texture,
            (self.image_width, self.image_height),
            interpolation=cv2.INTER_AREA,
        )

        return captured_view

    def pose_callback(self, msg):
        """
        Callback function to process the robot's pose and generate a camera image.
        """
        if (
            self.time is not None
            and self.get_clock().now() - self.time
            > rclpy.time.Duration(nanoseconds=1e9 / self.frequency)
        ):
            # Get the transform from map to base_link
            self.time = self.get_clock().now()
            transform = PoseStamped()
            transform.pose = msg.pose.pose

            # get rotation
            qx = transform.pose.orientation.x
            qy = transform.pose.orientation.y
            qz = transform.pose.orientation.z
            qw = transform.pose.orientation.w
            # Convert quaternion to rotation matrix
            r = R.from_quat([qx, qy, qz, qw])

            # Extract Euler angles (roll, pitch, yaw)
            roll, pitch, yaw = r.as_euler("xyz")

            # Robot's position in the map frame
            robot_x = transform.pose.position.x
            robot_y = transform.pose.position.y

            # Get the image
            image = self.simulate_gopro_view([robot_x, robot_y], yaw)

            # Publish the image
            try:
                image_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
                image_msg.header.stamp = msg.header.stamp
                image_msg.header.frame_id = (
                    "camera_frame"  # You might want to create a camera frame
                )
                self.image_pub.publish(image_msg)
            except Exception as e:
                self.get_logger().error(f"Error converting image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = VirtualGoProNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
