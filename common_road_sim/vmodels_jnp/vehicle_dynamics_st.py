import numpy as np
from numba import njit

from .vehicle_dynamics_ks_cog import vehicle_dynamics_ks_cog
from .steering_constraints import steering_constraints
from .acceleration_constraints import acceleration_constraints


@njit(cache=True, nopython=True, nogil=True)
def vehicle_dynamics_st(x, uInit, p):
    """
    vehicleDynamics_st - single-track vehicle dynamics
    reference point: center of mass

    Syntax:
        f = vehicleDynamics_st(x,u,p)

    Inputs:
        :param x: vehicle state vector
        :param uInit: vehicle input vector
        :param p: vehicle parameter vector

    Outputs:
        :return f: right-hand side of differential equations

    Author: Matthias Althoff
    Written: 12-January-2017
    Last update: 16-December-2017
                 03-September-2019
    Last revision: 17-November-2020
    """
    # set gravity constant
    g = 9.81  # [m/s^2]

    # create equivalent bicycle parameters
    mu = p["tire.p_dy1"]
    C_Sf = -p["tire.p_ky1"] / p["tire.p_dy1"]
    C_Sr = -p["tire.p_ky1"] / p["tire.p_dy1"]
    lf = p["a"]
    lr = p["b"]
    h = p["h_s"]
    m = p["m"]
    I = p["I_z"]

    # states
    # x1 = x-position in a global coordinate system
    # x2 = y-position in a global coordinate system
    # x3 = steering angle of front wheels
    # x4 = velocity in x-direction
    # x5 = yaw angle
    # x6 = yaw rate
    # x7 = slip angle at vehicle center

    # u1 = steering angle velocity of front wheels
    # u2 = longitudinal acceleration

    # consider steering constraints
    u = list()
    u.append(
        steering_constraints(x[2], uInit[0], p)
    )  # different name u_init/u due to side effects of u
    # consider acceleration constraints
    u.append(
        acceleration_constraints(x[3], uInit[1], p)
    )  # different name u_init/u due to side effects of u

    # switch to kinematic model for small velocities
    if abs(x[3]) < 0.1:
        # Use kinematic model with reference point at center of mass
        # wheelbase
        lwb = p["a"] + p["b"]
        # system dynamics
        x_ks = [x[0], x[1], x[2], x[3], x[4]]
        # kinematic model
        f_ks = vehicle_dynamics_ks_cog(x_ks, u, p)
        f = [f_ks[0], f_ks[1], f_ks[2], f_ks[3], f_ks[4]]
        # derivative of slip angle and yaw rate
        d_beta = (p["b"] * u[0]) / (
            lwb * np.cos(x[2]) ** 2 * (1 + (np.tan(x[2]) ** 2 * p["b"] / lwb) ** 2)
        )
        dd_psi = (
            1
            / lwb
            * (
                u[1] * np.cos(x[6]) * np.tan(x[2])
                - x[3] * np.sin(x[6]) * d_beta * np.tan(x[2])
                + x[3] * np.cos(x[6]) * u[0] / np.cos(x[2]) ** 2
            )
        )
        f.append(dd_psi)
        f.append(d_beta)

    else:
        # system dynamics
        f = [
            x[3] * np.cos(x[6] + x[4]),
            x[3] * np.sin(x[6] + x[4]),
            u[0],
            u[1],
            x[5],
            -mu
            * m
            / (x[3] * I * (lr + lf))
            * (lf**2 * C_Sf * (g * lr - u[1] * h) + lr**2 * C_Sr * (g * lf + u[1] * h))
            * x[5]
            + mu
            * m
            / (I * (lr + lf))
            * (lr * C_Sr * (g * lf + u[1] * h) - lf * C_Sf * (g * lr - u[1] * h))
            * x[6]
            + mu * m / (I * (lr + lf)) * lf * C_Sf * (g * lr - u[1] * h) * x[2],
            (
                mu
                / (x[3] ** 2 * (lr + lf))
                * (C_Sr * (g * lf + u[1] * h) * lr - C_Sf * (g * lr - u[1] * h) * lf)
                - 1
            )
            * x[5]
            - mu
            / (x[3] * (lr + lf))
            * (C_Sr * (g * lf + u[1] * h) + C_Sf * (g * lr - u[1] * h))
            * x[6]
            + mu / (x[3] * (lr + lf)) * (C_Sf * (g * lr - u[1] * h)) * x[2],
        ]

    return f
