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
import yaml

#!/usr/bin/env python3


class VirtualGoProNode(Node):
    def __init__(self):
        super().__init__("virtual_go_pro")

        # Declare parameters
        # self.declare_parameter("image_path", "../resource/clark_park/ClarkPark.jpg")

        # self.declare_parameter("camera_info_url", "")  # Optional: URL for camera info

        self.map_metadata = (
            get_package_share_directory("common_road_sim")
            + "/resource/clark_park/map_metadata.yaml"
        )

        # Parse the map metadata
        try:
            with open(self.map_metadata, "r") as f:
                metadata = yaml.safe_load(f)
                # These are the [u,v] pixel coordinates and accompanying [x,y] world coordinates of the reference pixel
                self.reference = (metadata["reference"][:2])
                self.reference_position = (metadata["reference_position"][:2]) # We only care about x and y

                self.pixels_per_meter = 1/metadata["resolution"] # pixels per meter instead of meters per pixel
                self.map_height = metadata["height"]
                self.map_width = metadata["width"]
                self.image_name = metadata["image"]
                self.get_logger().info(
                    f"{self.image_name} metadata: origin={self.reference_position}, reference={self.reference}, resolution={1/self.pixels_per_meter}, "
                    f"width={self.map_width}, height={self.map_height}"
                )
        except Exception as e:
            self.get_logger().error(f"Error parsing map metadata: {e}")
            rclpy.shutdown()
            return
        
        # Get parameters
        self.image_path = (
            get_package_share_directory("common_road_sim")
            + "/resource/clark_park/" + self.image_name
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
        self.camera_height = 2.5  # Height of the camera above the ground (in meters)
        self.camera_fov = 60.0  # Field of view in degrees
        self.footprint_size = self._fov_to_footprint(
            self.camera_height, self.camera_fov
        )

        self.image_width = 640
        self.image_height = 480

        self.get_logger().info("Virtual GoPro node initialized.")

    def _fov_to_footprint(self, height, fov_degrees, aspect_ratio=1.0):
        """
        Compute the footprint size on the ground given the camera's height, 
        vertical field-of-view, and sensor aspect ratio.
        
        Parameters:
        height       : The altitude of the camera in meters.
        fov_degrees  : The vertical field-of-view angle in degrees.
        aspect_ratio : The sensor aspect ratio (width / height). Default is 1.0 (square).
        
        Returns:
        footprint_size: (width, height) of the ground footprint in meters.
        """
        # Convert FOV from degrees to radians
        fov_rad = np.radians(fov_degrees)
        # Compute the vertical dimension of the footprint.
        footprint_height = 2 * height * np.tan(fov_rad / 2)
        # Compute the horizontal dimension using the aspect ratio.
        footprint_width = footprint_height * aspect_ratio
        return footprint_width, footprint_height
    
    def world_to_image_coords(self, x, y, scale, image_height, ref_pixel_coord, ref_pixel):
        """
        Convert world coordinates (meters) to image pixel coordinates.
        
        Parameters:
        x, y            : World coordinates in meters.
        scale           : Conversion factor (pixels per meter).
        image_height    : Height of the image in pixels (used to flip the y-axis).
        ref_pixel_coord : (origin_x, origin_y) location of the image's reference pixel in world coordinates.
        ref_pixel       : (u, v) location of the image's reference pixel in pixel coordinates.
        
        Returns:
        (px, py)     : Pixel coordinates in the image.
        """
        print(f"scale: {scale} , image_height: {image_height} , ref_pixel_coord: {ref_pixel_coord} , ref_pixel: {ref_pixel}", flush=True)
        px = ref_pixel[0] + scale * (x - ref_pixel_coord[0])
        py = image_height - (ref_pixel[1] + scale * (y - ref_pixel_coord[1]))
        return int(px), int(py)

    def simulate_gopro_view(self, large_image, gopro_pos, footprint_size, scale, reference_pix, reference_pos, angle=0):
        """
        Extract the gopro's view from the large image.
        
        Parameters:
        large_image   : Numpy array of the large image.
        vehicle_pose  : (x, y) gopro's center position in world coordinates (meters).
        footprint_size: (width, height) in meters of the camera's field of view.
        scale         : Conversion factor (pixels per meter).
        reference_pix : (u, v) location of the image's reference pixel in pixel coordinates.
        reference_pos : (origin_x, origin_y) reference pixel location in world coordinates.
        angle         : Rotation angle in degrees (yaw) for the camera.
        
        Returns:
        captured_view: The sub-image corresponding to the gopro's current view.
        """
        img_h, img_w = large_image.shape[:2]
        
        # Convert gopro's world position to image pixel coordinates.
        center_px, center_py = self.world_to_image_coords(gopro_pos[0], gopro_pos[1], scale, img_h, reference_pos, reference_pix)
        
        half_width_px = (footprint_size[0] * scale) / 2
        half_height_px = (footprint_size[1] * scale) / 2
        
        # For zero rotation, crop a simple rectangle.
        if angle % 360 == 0:
            left = int(center_px - half_width_px)
            right = int(center_px + half_width_px)
            top = int(center_py - half_height_px)
            bottom = int(center_py + half_height_px)
            
            left = max(left, 0)
            right = min(right, img_w)
            top = max(top, 0)
            bottom = min(bottom, img_h)
            
            captured_view = large_image[top:bottom, left:right].copy()
        else:
            # Define the rotated rectangle.
            rect = ((center_px, center_py), (footprint_size[0] * scale, footprint_size[1] * scale), -angle)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            width = int(footprint_size[0] * scale)
            height = int(footprint_size[1] * scale)
            dst_pts = np.array([[0, height - 1],
                                [0, 0],
                                [width - 1, 0],
                                [width - 1, height - 1]], dtype="float32")
            src_pts = box.astype("float32")
            
            # Compute the perspective transform and extract the view.
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            captured_view = cv2.warpPerspective(large_image, M, (width, height))
        
        return captured_view

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

        # Get the image
        image = self.simulate_gopro_view(self.full_map, 
                                         [robot_x, robot_y], 
                                         self.footprint_size, 
                                         self.pixels_per_meter,
                                         self.reference,
                                         self.reference_position,
                                         yaw)


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
