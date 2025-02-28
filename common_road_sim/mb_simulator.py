#!/usr/bin/env python3
import numpy as np
from vehiclemodels.init_mb import init_mb
from vehiclemodels.vehicle_dynamics_mb import vehicle_dynamics_mb
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2
from vehiclemodels.parameters_vehicle3 import parameters_vehicle3

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Imu
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from scipy.integrate import odeint
from scipy.spatial.transform import Rotation as R
from threading import Lock

"""
This module contains the class MBVehicleSimulator, which is a simulator for a multi-body vehicle model.
It runs on a clock at 100Hz using SciPy's odeint integration.
The model used is the multi-body model from the CommonRoad vehicle models.
The vehicle starts at the origin, with a heading of 0 degrees, and a velocity of 0 m/s.
The control inputs are:
    - u1 = steering angle velocity of front wheels
    - u2 = acceleration
"""

def pid_steer(steer, current_steer, max_sv):
    steer_diff = steer - current_steer
    if np.fabs(steer_diff) > 1e-4:
        return (steer_diff / np.fabs(steer_diff)) * max_sv
    else:
        return 0.0

def pid_accl(speed, current_speed, max_a, max_v, min_v):
    vel_diff = speed - current_speed
    FS_MOD = 1.5  # Modifier for Full Scale vehicles.
    if current_speed > 0.0:
        if vel_diff > 0:
            kp = FS_MOD * 10.0 * max_a / max_v
            return kp * vel_diff
        else:
            kp = FS_MOD * 10.0 * max_a / (-min_v)
            return kp * vel_diff
    else:
        if vel_diff > 0:
            kp = FS_MOD * 2.0 * max_a / max_v
            return kp * vel_diff
        else:
            kp = FS_MOD * 2.0 * max_a / (-min_v)
            return kp * vel_diff

def integrate_model(state, control_input, parameters, dt):
    """
    Integrate the vehicle dynamics using SciPy's odeint over the interval [0, dt].
    """
    def model_dynamics(x, t, u, p):
        return vehicle_dynamics_mb(x, u, p)
    t_span = [0, dt]
    # Tolerances may need tuning for your dynamics.
    next_state = odeint(model_dynamics, state, t_span, args=(control_input, parameters), rtol=1e-3, atol=1e-3)
    return next_state[-1]

class MBSimulator(Node):
    def __init__(self):
        super().__init__("mb_simulator")
        self.declare_parameters(
            namespace="mb_simulator",
            parameters=[
                ("model", 1),
                ("frequency", 100),  # Target 100Hz simulation frequency.
            ],
        )
        self.freq = self.get_parameter("mb_simulator.frequency").get_parameter_value().integer_value
        self.model = self.get_parameter("mb_simulator.model").get_parameter_value().integer_value
        
        if self.model == 1:
            self.parameters = parameters_vehicle1()
        elif self.model == 2:
            self.parameters = parameters_vehicle2()
        elif self.model == 3:
            self.parameters = parameters_vehicle3()
        else:
            raise ValueError("Invalid model selected, please select 1, 2, or 3")
        
        # Initialize state vector; here we assume a 7-dimensional state.
        initial_state = np.array([0, 0, 0, 0, 0, 0, 0])
        self.state = init_mb(initial_state, self.parameters)
        # Initialize with a nonzero control input so the vehicle can move.
        self.control_input = np.array([1.0, 1.0])
        
        control_cbg = MutuallyExclusiveCallbackGroup()
        dynamics_cbg = MutuallyExclusiveCallbackGroup()
        
        self.timer = self.create_timer(1.0 / self.freq, self.timer_callback, callback_group=dynamics_cbg)
        self.odom_pub = self.create_publisher(Odometry, "/fixposition/odometry", 10)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, "/ground_truth/pose", 10)
        self.imu_pub = self.create_publisher(Imu, "/fixposition/corrimu", 10)
        self.control_sub = self.create_subscription(
            AckermannDriveStamped,
            "/drive",
            self.steer_callback,
            1,
            callback_group=control_cbg,
        )
        self.control_lock = Lock()

    def timer_callback(self):
        with self.control_lock:
            control_input = self.control_input.copy()
        self.get_logger().debug(f"Control input: {control_input}")
        dt = 1.0 / self.freq
        self.state = integrate_model(self.state, control_input, self.parameters, dt)
        self.publish_pose_and_covariance(control_input)

    def steer_callback(self, msg):
        steerv = pid_steer(
            msg.drive.steering_angle, self.state[2], self.parameters.steering.v_max
        )
        accl = pid_accl(
            msg.drive.speed,
            self.state[3],
            self.parameters.longitudinal.a_max,
            self.parameters.longitudinal.v_max,
            self.parameters.longitudinal.v_min,
        )
        with self.control_lock:
            self.control_input = np.array([steerv, accl])
        self.get_logger().debug(f"Steering: {steerv}, Acceleration: {accl}")

    def publish_pose_and_covariance(self, control_input=None):
        if control_input is None:
            control_input = self.control_input
        pose_message = PoseWithCovarianceStamped()
        pose_message.header.stamp = self.get_clock().now().to_msg()
        pose_message.header.frame_id = "map"
        pose_message.pose.pose.position.x = self.state[0]
        pose_message.pose.pose.position.y = self.state[1]
        # Use state[11] for z, if available.
        pose_message.pose.pose.position.z = self.state[11] if len(self.state) > 11 else 0.0
        
        # Convert Euler (z rotation) to quaternion.
        r = R.from_euler("z", self.state[4], degrees=False)
        q = r.as_quat()  # q is [x, y, z, w]
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q /= q_norm
        # Publish quaternion in ROS order: x, y, z, w.
        pose_message.pose.pose.orientation.x = q[0]
        pose_message.pose.pose.orientation.y = q[1]
        pose_message.pose.pose.orientation.z = q[2]
        pose_message.pose.pose.orientation.w = q[3]

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

        imu_message = Imu()
        imu_message.header = pose_message.header
        imu_message.orientation = pose_message.pose.pose.orientation
        imu_message.angular_velocity.x = self.state[7]
        imu_message.angular_velocity.y = self.state[9]
        imu_message.angular_velocity.z = self.state[5]
        imu_message.linear_acceleration.x = control_input[1]

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
