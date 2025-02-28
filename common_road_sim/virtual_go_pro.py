import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import griddata
import traceback
import yaml

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
        self.image_meta_path = (
            get_package_share_directory("common_road_sim")
            + "/resource/clark_park/map_metadata.yaml"
        )
        # Load metadata
        try:
            with open(self.image_meta_path, "r") as file:
                self.metadata = yaml.safe_load(file)
            self.get_logger().info(f"Loaded map metadata successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to load metadata: {str(e)}")
            self.metadata = {}

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
        self.camera_fov = np.radians(60.0)  # Field of view in radians
        self.image_width = 640
        self.image_height = 480
        self.aspect_ratio = self.image_width / self.image_height
        self.pixels_per_meter = (
            1.0 / self.metadata["resolution"]
        )  # Adjust this based on your map's scale

        # The camera is mounted at a fixed height above the ground, facing directly downwards.
        diagonal = self.camera_height * np.tan(self.camera_fov / 2.0)
        self.image_width_meters = np.sqrt(
            (4 * diagonal**2) / (1 + self.aspect_ratio**2)
        )
        self.image_height_meters = self.aspect_ratio * self.image_width_meters

        # Relate each pixel value to a location on the map
        self.map_coords, self.image_coords, self.map_coord_image = self.convert_base_image_to_map_coordinates(
            self.full_map
        )

        # Print information about map coordinates
        self.get_logger().info(f"Map coordinates range: X from {np.min(self.map_coords[:,0]):.2f} to {np.max(self.map_coords[:,0]):.2f} meters")
        self.get_logger().info(f"Map coordinates range: Y from {np.min(self.map_coords[:,1]):.2f} to {np.max(self.map_coords[:,1]):.2f} meters")
        self.get_logger().info(f"Pixels per meter: {self.pixels_per_meter:.2f}")

        self.get_logger().info("Virtual GoPro node initialized.")

    def convert_base_image_to_map_coordinates(self, image):
        """
        Convert the base image to map coordinates.
        Instead of (H,W,C), we want (x,y,C) where x and y are in meters.
        """
        # Create a meshgrid for the image
        x = np.linspace(0, self.map_width-1, self.map_width)
        y = np.linspace(0, self.map_height-1, self.map_height)
        xx, yy = np.meshgrid(x, y)

        # Convert the image to map coordinates
        map_x = (xx - self.map_width / 2) / self.pixels_per_meter - self.metadata["origin"][0]
        map_y = (yy - self.map_height / 2) / self.pixels_per_meter - self.metadata["origin"][1]

        # Save the map_coordinates as a 2d array, Npoints x 2
        map_coords = np.stack([map_x.flatten(), map_y.flatten()], axis=1)
        image_coords = np.stack([xx.flatten(), yy.flatten()], axis=1)

        # Save the map_image as a 2d array, Npoints x 3, where the points correspond to the map_coords using the image_coords
        map_coord_image = np.zeros((map_coords.shape[0], 3), dtype=np.uint8)
        for i in range(map_coords.shape[0]):
            x = int(image_coords[i, 0])
            y = int(image_coords[i, 1])
            map_coord_image[i] = image[y, x]
        
        return map_coords, image_coords, map_coord_image


    def publish_image(self, image):
        """
        Publish the image to the camera/image_raw topic.
        """
        try:
            image_msg = self.bridge.cv2_to_imgmsg(image)
            image_msg.header.stamp = self.get_clock().now().to_msg()
            image_msg.header.frame_id = "camera_frame"
            self.image_pub.publish(image_msg)
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

    def get_image_by_box(self, bottom_left, bottom_right, top_left, top_right):
        """
        Get the image by the bounding box using interpolation.
        Note that the coordinates must be valid.

        Args:
            bottom_left: tuple (x, y) - Bottom-left corner of the bounding box
            bottom_right: tuple (x, y) - Bottom-right corner of the bounding box
            top_left: tuple (x, y) - Top-left corner of the bounding box
            top_right: tuple (x, y) - Top-right corner of the bounding box

        Returns:
            Image within the specified bounding box
        """
        points = np.array([
            [0,0],  # Bottom Left
            [0,1],  # Top Left
            [1,0],  # Bottom Right
            [1,1]   # Top Right
        ])
        values = np.array([
            bottom_left,
            top_left,
            bottom_right,
            top_right
        ])
        x = np.linspace(0, 1, self.image_width)
        y = np.linspace(0, 1, self.image_height)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.stack([xx.flatten(), yy.flatten()], axis=1)
        image_coords_meters = griddata(points, values, grid_points, method="linear")
        self.get_logger().info(f"Interpolated image coordinates")

        # For the image coords, get the pixel values:
        image_data = griddata(self.map_coords, self.map_coord_image, image_coords_meters, method="linear", fill_value=0)
        image_data = image_data.reshape(self.image_height, self.image_width, 3)

        self.get_logger().info(f"Interpolated image data")
        return image_data

    def pose_callback(self, msg):
        """
        Callback function to process the robot's pose and generate a camera image.
        """
        self.get_logger().info("Received pose message.")
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

        # The camera is mounted at a fixed height above the ground, facing directly downwards.
        top_left = (
            robot_x - self.image_width_meters / 2,
            robot_y + self.image_height_meters / 2,
        )
        top_right = (
            robot_x + self.image_width_meters / 2,
            robot_y + self.image_height_meters / 2,
        )
        bottom_left = (
            robot_x - self.image_width_meters / 2,
            robot_y - self.image_height_meters / 2,
        )
        bottom_right = (
            robot_x + self.image_width_meters / 2,
            robot_y - self.image_height_meters / 2,
        )
        self.get_logger().info(f"Obtained bounding box: {top_left}, {top_right}, {bottom_left}, {bottom_right}")
        image = self.get_image_by_box(bottom_left, bottom_right, top_left, top_right)
        # save the image
        cv2.imwrite("/FrictionEstimation/ros/src/common_road_sim/common_road_sim/virtual_gopro_image.jpg", image)
        self.publish_image(image)


def main(args=None):
    rclpy.init(args=args)
    node = VirtualGoProNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
