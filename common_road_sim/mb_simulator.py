from .vmodels_jnp.init_ks import init_ks
from .vmodels_jnp.parameters import (
    parameters_vehicle1,
    parameters_vehicle2,
    parameters_vehicle3,
)
from .vmodels_jnp.init_mb import init_mb
from .vmodels_jnp.vehicle_dynamics_mb import vehicle_dynamics_mb
from .vmodels_jnp.vehicle_dynamics_st import vehicle_dynamics_st

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32
import numpy as np

from numba import njit
from scipy.integrate import odeint, solve_ivp
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from scipy.integrate import odeint
from scipy.spatial.transform import Rotation as R
from threading import Lock

try:
    from context_msgs.msg import STControl, STState, STCombined

    GT_STATE_PUB = True
except ImportError:
    GT_STATE_PUB = False
    print("context_msgs not found, GT_STATE_PUB will not be used.")

MB_VEHICLE_MODEL = False

"""
This module contains the class MBVehicleSimulator, which is a simulator for a multi-body vehicle model.
It runs on a clock at 100Hz using SciPy's odeint integration.
The model used is the multi-body model from the CommonRoad vehicle models.
The vehicle starts at the origin, with a heading of 0 degrees, and a velocity of 0 m/s.
The control inputs are:
    - u1 = steering angle velocity of front wheels
    - u2 = acceleration
"""


@njit(cache=True)
def pid_steer(steer, current_steer, max_sv, dt):
    error = steer - current_steer
    steer_rate = error / dt
    if steer_rate > max_sv:
        steer_rate = max_sv
    elif steer_rate < -max_sv:
        steer_rate = -max_sv
    return steer_rate


@njit(cache=True)
def pid_accl(speed, current_speed, max_a, max_v, min_v, integral_error, dt):
    """
    Proportional-Integral controller for acceleration.
    dt: time step duration.
    """
    vel_diff = speed - current_speed
    FS_MOD = 3.0  # Modifier for Full Scale vehicles.

    if current_speed > 0.0:
        if vel_diff > 0:
            kp = FS_MOD * 10.0 * max_a / max_v
            ki = 0.1 * kp  # Integral gain
        else:
            kp = FS_MOD * 10.0 * max_a / (-min_v)
            ki = 0.1 * kp
    else:
        if vel_diff > 0:
            kp = FS_MOD * 10.0 * max_a / max_v
            ki = 0.1 * kp
        else:
            kp = FS_MOD * 10.0 * max_a / (-min_v)
            ki = 0.1 * kp

    # Accumulate error using a static (function attribute) variable.
    integral_error += vel_diff * dt

    control_signal = kp * vel_diff + ki * integral_error
    return integral_error, control_signal


def integrate_model(state, control_input, parameters, dt):
    """
    Integrate the vehicle dynamics using SciPy's odeint over the interval [0, dt].
    """

    def model_dynamics(x, t, u, p):
        if MB_VEHICLE_MODEL:
            return vehicle_dynamics_mb(x, u, p)
        else:
            return vehicle_dynamics_st(x, u, p)

    t_span = [0, dt]
    next_state = odeint(
        model_dynamics,
        state,
        t_span,
        args=(control_input, parameters),
        rtol=1e-3,
        atol=1e-3,
    )[-1]

    return next_state


def transform_yaw(yaw):
    """
    Transform the yaw angle to be within the range [0, 2*pi].
    """
    return yaw % (2 * np.pi)


class MBSimulator(Node):
    def __init__(self, ground_truth_pub=True):
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

        # Initialize state vector; here we assume a 7-dimensional state.
        # X = [x, y, delta, v, yaw, yaw_rate, beta]
        initial_state = np.array([12, 0, 0, 0, -1.5, 0, 0])
        if MB_VEHICLE_MODEL:
            self.state = init_mb(initial_state, self.parameters)
        else:
            self.state = initial_state  # For Single Track Model
        # Initialize with a nonzero control input so the vehicle can move.
        self.control_input = np.array([0.0, 0.0])

        control_cbg = MutuallyExclusiveCallbackGroup()
        dynamics_cbg = MutuallyExclusiveCallbackGroup()
        friction_cbg = MutuallyExclusiveCallbackGroup()

        self.timer = self.create_timer(
            1.0 / self.freq, self.timer_callback, callback_group=dynamics_cbg
        )
        self.odom_pub = self.create_publisher(Odometry, "/fixposition/odometry", 10)
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/ground_truth/pose", 10
        )
        self.imu_pub = self.create_publisher(Imu, "/fixposition/corrimu", 10)
        self.control_sub = self.create_subscription(
            AckermannDriveStamped,
            "/drive",
            self.steer_callback,
            1,
            callback_group=control_cbg,
        )
        self.friction_sub = self.create_subscription(
            Float32,
            "/friction",
            self.friction_callback,
            1,
            callback_group=friction_cbg,
        )
        self.ground_truth_pub = ground_truth_pub
        if ground_truth_pub:
            self.gt_state_pub = self.create_publisher(
                STState, "/ground_truth/state", 10
            )
            self.gt_control_pub = self.create_publisher(
                STControl, "/ground_truth/control", 10
            )
            self.gt_combined_pub = self.create_publisher(
                STCombined, "/ground_truth/combined", 10
            )

        self.control_lock = Lock()
        self.parameter_lock = Lock()

        self.get_logger().info(
            "MB Vehicle Simulator running at 100Hz with odeint integration."
        )

        self.speed_integral_error = 0.0
        self.last_control_command_time = self.get_clock().now()

    def timer_callback(self):
        """This function updates the state of the vehicle and publishes the ground truth odometry and pose."""
        with self.control_lock:
            control_input = self.control_input.copy()
        # self.get_logger().info(
        #     f"Friction: {self.parameters['tire.p_dy1']} | Control input: {control_input}"
        # )
        with self.parameter_lock:
            self.state = integrate_model(
                self.state, control_input, self.parameters, 1 / self.freq
            )
        self.publish_pose_and_covariance(control_input)

        if self.ground_truth_pub:
            self.publish_gt_state(control_input)

    def publish_gt_state(self, control_input):
        stamp = self.get_clock().now().to_msg()
        state_message = STState()
        control_message = STControl()
        combined_message = STCombined()
        state_message.header.stamp = stamp
        control_message.header.stamp = stamp
        combined_message.header.stamp = stamp

        if MB_VEHICLE_MODEL:
            state_message.x = self.state[0]
            state_message.y = self.state[1]
            state_message.velocity = self.state[3]
            state_message.yaw = transform_yaw(self.state[4])
            state_message.yaw_rate = self.state[5]
            state_message.slip_angle = np.arctan2(self.state[10], self.state[3])
            control_message.steering_angle = self.state[2]
            control_message.acceleration = control_input[1]
        else:
            # SINGLE TRACK MODEL
            state_message.x = self.state[0]
            state_message.y = self.state[1]
            state_message.velocity = self.state[3]
            state_message.yaw = transform_yaw(self.state[4])
            state_message.yaw_rate = self.state[5]
            state_message.slip_angle = self.state[6]
            control_message.steering_angle = self.state[2]
            control_message.acceleration = control_input[1]

        combined_message.state = state_message
        combined_message.control = control_message
        self.gt_state_pub.publish(state_message)
        self.gt_control_pub.publish(control_message)
        self.gt_combined_pub.publish(combined_message)

    def steer_callback(self, msg):
        new_time = self.get_clock().now()
        dt = 1 / self.freq
        # Check if we get the steering velocity or the steering angle:

        if msg.drive.steering_angle == 0.0:
            # If the steering angle is 0, we assume it's a steering velocity.
            # Convert to steering angle.
            steerv = msg.drive.steering_angle_velocity
        else:
            # If the steering angle is not 0, we assume it's a steering angle.
            # Convert to steering velocity.
            steerv = pid_steer(
                msg.drive.steering_angle,
                self.state[2],
                self.parameters["steering.v_max"],
                dt,
            )

        if msg.drive.speed == 0.0:
            # If the speed is 0, we assume it's an acceleration.
            # Convert to speed.
            accl = msg.drive.acceleration
        else:
            self.speed_integral_error, accl = pid_accl(
                msg.drive.speed,
                self.state[3],
                self.parameters["longitudinal.a_max"],
                self.parameters["longitudinal.v_max"],
                self.parameters["longitudinal.v_min"],
                self.speed_integral_error,
                dt,
            )
            self.last_control_command_time = new_time

        if np.isnan(steerv):
            steerv = 0.0
        if np.isnan(accl):
            accl = 0.0

        self.control_lock.acquire()
        self.control_input = np.array([steerv, accl])
        self.control_lock.release()
        # self.get_logger().info(f"Steering: {steerv}, Acceleration: {accl}")

    def friction_callback(self, msg):
        with self.parameter_lock:
            self.parameters["tire.p_dy1"] = msg.data
        # self.get_logger().info(f"Friction coefficient updated to {msg.data}")

    def publish_pose_and_covariance(self, control_input=None):
        if control_input is None:
            control_input = self.control_input
        pose_message = PoseWithCovarianceStamped()
        pose_message.header.stamp = self.get_clock().now().to_msg()
        pose_message.header.frame_id = "map"
        pose_message.pose.pose.position.x = self.state[0]
        pose_message.pose.pose.position.y = self.state[1]
        # Use state[11] for z, if available.
        pose_message.pose.pose.position.z = (
            self.state[11] if len(self.state) > 11 else 0.0
        )

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
        if MB_VEHICLE_MODEL:
            slip_angle = np.arctan2(self.state[10], self.state[3])
            twist_message.twist.twist.linear.z = self.state[12]
            twist_message.twist.twist.angular.x = self.state[9]
            twist_message.twist.twist.angular.y = self.state[7]
        else:
            slip_angle = self.state[6]
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
        if MB_VEHICLE_MODEL:
            imu_message.angular_velocity.x = self.state[9]
            imu_message.angular_velocity.y = self.state[7]
        imu_message.angular_velocity.z = self.state[5]
        imu_message.linear_acceleration.x = control_input[1]

        self.imu_pub.publish(imu_message)
        self.pose_pub.publish(pose_message)
        self.odom_pub.publish(odom_message)


def main(args=None):
    rclpy.init(args=args)
    simulator = MBSimulator(ground_truth_pub=GT_STATE_PUB)
    rclpy.spin(simulator)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
