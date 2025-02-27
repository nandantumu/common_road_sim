from vehiclemodels.init_ks import init_ks
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2
from vehiclemodels.parameters_vehicle3 import parameters_vehicle3
from vehiclemodels.init_mb import init_mb
from vehiclemodels.vehicle_dynamics_mb import vehicle_dynamics_mb

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from sensor_msgs.msg import Imu
import numpy as np
from scipy.integrate import odeint
from rclpy.node import Node
import rclpy
from scipy.spatial.transform import Rotation as R
from threading import Lock

"""
This module contains the class MBVehicleSimulator, which is a simulator for a multi-body vehicle model. It runs on a clock at 100Hz, using scipy odeint to integrate the model forward. The model used is the multi-body model from the CommonRoad vehicle models. The simulator is initialized with a vehicle model, the user can select 1, 2, or 3. The vehicle starts at the origin, with a heading of 0 degrees, and a velocity of 0 m/s. The simulator can be controlled by setting the steering angle and the throttle.

This node publishes the following topics:
 - /ground_truth/odometry (Odometry): The ground truth odometry of the vehicle, including the position, velocity, and heading.
 - /ground_truth/pose (PoseWithCovarianceStamped): The ground truth pose of the vehicle, including the position and heading.

The control inputs are:
    - u1 = steering angle velocity of front wheels
    - u2 = acceleration

"""


def integrate_model(state, control_input, parameters, dt=0.01):
    def model_dynamics(x, t, u, p):
        return vehicle_dynamics_mb(x, u, p)

    # Integrate the model from t=0 to t=dt
    t_span = [0, dt]
    next_state = odeint(model_dynamics, state, t_span, args=(control_input, parameters))
    return next_state[-1]


class MBSimulator(Node):
    def __init__(self):
        super().__init__("mb_simulator")
        self.declare_parameters(
            namespace="mb_simulator",
            parameters=[
                ("model", 1),
                ("frequency", 100),
            ],
        )
        self.freq = (
            self.get_parameter("mb_simulator.frequency")
            .get_parameter_value()
            .integer_value
        )
        self.model = (
            self.get_parameter("mb_simulator.model").get_parameter_value().integer_value
        )
        if self.model == 1:
            self.parameters = parameters_vehicle1()
        elif self.model == 2:
            self.parameters = parameters_vehicle2()
        elif self.model == 3:
            self.parameters = parameters_vehicle3()
        else:
            raise ValueError("Invalid model selected, please select 1, 2, or 3")
        initial_state = np.array([0, 0, 0, 0, 0, 0, 0])
        self.state = init_mb(initial_state, self.parameters)
        self.control_input = np.array([-0.15, 2 * 9.81])
        self.timer = self.create_timer(1.0 / self.freq, self.timer_callback)
        self.odom_pub = self.create_publisher(Odometry, "/fixposition/odometry", 10)
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/ground_truth/pose", 10
        )
        self.imu_pub = self.create_publisher(Imu, "/fixposition/corrimu", 10)
        self.control_sub = self.create_subscription(
            AckermannDriveStamped, "/ackermann_cmd", self.steer_callback, 10
        )
        self.control_lock = Lock()

    def timer_callback(self):
        """This function updates the state of the vehicle and publishes the ground truth odometry and pose."""
        self.control_lock.acquire()
        self.state = integrate_model(
            self.state, self.control_input, self.parameters, 1 / self.freq
        )
        self.publish_pose_and_covariance()
        # print("x:", self.state[0], "y:", self.state[1])
        self.control_lock.release()

    def steer_callback(self, msg):
        """This function sets the control input based on the steering angle and throttle."""
        self.control_lock.acquire()
        self.control_input = np.array(
            [msg.drive.steering_angle_velocity, msg.drive.acceleration]
        )
        self.control_lock.release()

    def publish_pose_and_covariance(self):
        """This function publishes the ground truth pose of the vehicle."""
        pose_message = PoseWithCovarianceStamped()
        pose_message.header.stamp = self.get_clock().now().to_msg()
        pose_message.header.frame_id = "map"
        pose_message.pose.pose.position.x = self.state[0]
        pose_message.pose.pose.position.y = self.state[1]
        pose_message.pose.pose.position.z = self.state[11]
        r = R.from_euler("z", self.state[4], degrees=False)
        q = r.as_quat()
        pose_message.pose.pose.orientation.w = q[0]
        pose_message.pose.pose.orientation.x = q[1]
        pose_message.pose.pose.orientation.y = q[2]
        pose_message.pose.pose.orientation.z = q[3]

        twist_message = TwistWithCovarianceStamped()
        twist_message.header = pose_message.header
        slip_angle = np.arctan2(self.state[10], self.state[3])
        twist_message.twist.twist.linear.x = self.state[3] * np.cos(slip_angle)
        twist_message.twist.twist.linear.y = self.state[3] * np.sin(slip_angle)
        twist_message.twist.twist.angular.z = self.state[5]

        odom_message = Odometry()
        odom_message.header = pose_message.header
        odom_message.pose = pose_message.pose
        odom_message.twist = twist_message.twist

        # Create the Imu message:
        imu_message = Imu()
        imu_message.header = pose_message.header
        imu_message.orientation = pose_message.pose.pose.orientation
        imu_message.angular_velocity.x = self.state[7]
        imu_message.angular_velocity.y = self.state[9]
        imu_message.angular_velocity.z = self.state[5]
        imu_message.linear_acceleration.x = self.control_input[1]

        self.imu_pub.publish(imu_message)
        self.pose_pub.publish(pose_message)
        self.odom_pub.publish(odom_message)


def main(args=None):
    rclpy.init(args=args)
    simulator = MBSimulator()
    rclpy.spin(simulator)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
