import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.transform import Rotation as R
import traceback

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

    def image_rotation(self, image, point, angle):
        """
        Rotate the image by the specified angle.
        """
        rot_matrix = cv2.getRotationMatrix2D(point, angle, 1.0)
        image_out = cv2.warpAffine(image, rot_matrix, (self.map_width, self.map_height))
        return image_out

    def pose_callback(self, msg):
        """
        Callback function to process the robot's pose and generate a camera image.
        """
        # Get the transform from map to base_link
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

        # Calculate the image center in map coordinates
        center_x = int(robot_x * self.pixels_per_meter)
        center_y = int(robot_y * self.pixels_per_meter)
        point = (center_x, center_y)

        # Calculate the radius of the image
        radius = np.sqrt((self.image_width / 2) ** 2 + (self.image_height / 2) ** 2)
        radius = int(
            radius * 1.2
        )  # Increase the radius to ensure the entire image is visible
        print("Radius:", radius)

        try:
            y_low = center_y - radius
            y_high = center_y + radius
            x_low = center_x - radius
            x_high = center_x + radius
            print("Indices:", y_low, y_high, x_low, x_high)
            if (
                y_low < 0
                or y_high > self.map_height
                or x_low < 0
                or x_high > self.map_width
            ):
                raise IndexError("Image subset is out of bounds")
            else:
                image_subset = self.full_map[
                    center_y - radius : center_y + radius,
                    center_x - radius : center_x + radius,
                ]
        except IndexError as e:
            # Likely problem is that the image subset is out of bounds
            # We will solve this problem by padding the image with zeros
            image_subset = np.zeros((2 * radius, 2 * radius, 3), dtype=np.uint8)
            # Cast the image_subset to a cv2 image
            image_subset = cv2.cvtColor(image_subset, cv2.COLOR_RGB2BGR)
            x_min_valid = max(0, center_x - radius)
            y_min_valid = max(0, center_y - radius)
            x_max_valid = min(self.map_width, center_x + radius)
            y_max_valid = min(self.map_height, center_y + radius)

            # Only extract and assign if we have a valid region
            if y_min_valid < y_max_valid and x_min_valid < x_max_valid:
                valid_image = self.full_map[
                    y_min_valid:y_max_valid, x_min_valid:x_max_valid
                ]
                x_offset = x_min_valid - (center_x - radius)
                y_offset = y_min_valid - (center_y - radius)

                # Check that source and target dimensions are valid
                if valid_image.shape[0] > 0 and valid_image.shape[1] > 0:
                    image_subset[
                        y_offset : y_offset + valid_image.shape[0],
                        x_offset : x_offset + valid_image.shape[1],
                    ] = valid_image

        rotated_image = self.image_rotation(image_subset, (radius, radius), yaw)

        # Calculate the boundaries of the image
        center_x = radius
        center_y = radius
        x_min = center_x - self.image_width // 2
        y_min = center_y - self.image_height // 2
        x_max = center_x + self.image_width // 2
        y_max = center_y + self.image_height // 2

        # Extract the image from the rotated portion
        image = rotated_image[y_min:y_max, x_min:x_max]

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
