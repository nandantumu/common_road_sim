import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from scipy.spatial.transform import Rotation as R

#!/usr/bin/env python3


class CircleController(Node):
    def __init__(self):
        super().__init__("circle_controller")

        # Parameters for the circular motion
        self.circle_radius = 5.0  # meters
        self.linear_velocity = 2.0  # m/s

        # Create publisher for the drive commands
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped, "/drive", 10
        )

        # Create subscriber for the vehicle odometry
        self.odom_subscription = self.create_subscription(
            Odometry, "/fixposition/odometry", self.odom_callback, 10
        )

        self.get_logger().info("Circle Controller has been started")

    def odom_callback(self, msg):
        # Get current position
        current_x = msg.pose.pose.position.x
        current_y = msg.pose.pose.position.y

        # Get current orientation
        orientation_q = msg.pose.pose.orientation
        r = R.from_quat(
            [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        )
        _, _, yaw = r.as_euler("xyz")

        # Calculate distance from origin
        distance_from_origin = math.sqrt(current_x**2 + current_y**2)

        # Calculate angle from origin to current position
        angle_to_origin = math.atan2(-current_y, -current_x)

        # Calculate heading error (difference between current heading and tangent to the circle)
        desired_heading = angle_to_origin + math.pi / 2  # Tangent to the circle
        heading_error = self.normalize_angle(desired_heading - yaw)

        # Calculate steering angle (proportional to heading error and distance error)
        k_heading = 1.0  # Gain for heading error
        k_distance = 0.5  # Gain for distance error
        distance_error = distance_from_origin - self.circle_radius

        # Calculate steering angle using a simple proportional controller
        steering_angle = k_heading * heading_error + k_distance * distance_error
        steering_angle = max(min(steering_angle, 0.5), -0.5)  # Limit steering angle

        # Create and publish the drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.steering_angle_velocity = 0.0
        drive_msg.drive.speed = self.linear_velocity
        drive_msg.drive.acceleration = 0.0
        drive_msg.drive.jerk = 0.0

        self.drive_publisher.publish(drive_msg)
        self.get_logger().info(f"Steering angle: {steering_angle}")
        self.get_logger().info(f"Speed: {self.linear_velocity}")
        self.get_logger().info(f"Distance from origin: {distance_from_origin}")
        self.get_logger().info(f"Yaw: {yaw}")
        self.get_logger().info(f"Desired heading: {desired_heading}")

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    controller = CircleController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
