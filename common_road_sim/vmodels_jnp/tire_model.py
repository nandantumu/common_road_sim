import numpy as jnp
from numba import njit


# # sign function
# def sign(x):
#     return jnp.sign(x)


# longitudinal tire forces
@njit(cache=True)
def formula_longitudinal(kappa, gamma, F_z, p):
    # turn slip is neglected, so xi_i=1
    # all scaling factors lambda = 1

    # coordinate system transformation
    kappa = -kappa

    S_hx = p["tire.p_hx1"]
    S_vx = F_z * p["tire.p_vx1"]

    kappa_x = kappa + S_hx
    mu_x = p["tire.p_dx1"] * (1 - p["tire.p_dx3"] * gamma**2)

    C_x = p["tire.p_cx1"]
    D_x = mu_x * F_z
    E_x = p["tire.p_ex1"]
    K_x = F_z * p["tire.p_kx1"]
    B_x = K_x / (C_x * D_x)

    # magic tire formula
    return D_x * jnp.sin(
        C_x
        * jnp.arctan(B_x * kappa_x - E_x * (B_x * kappa_x - jnp.arctan(B_x * kappa_x)))
        + S_vx
    )


# lateral tire forces
@njit(cache=True)
def formula_lateral(alpha, gamma, F_z, p):
    # turn slip is neglected, so xi_i=1
    # all scaling factors lambda = 1

    # coordinate system transformation
    # alpha = -alpha

    S_hy = jnp.sign(gamma) * (p["tire.p_hy1"] + p["tire.p_hy3"] * jnp.fabs(gamma))
    S_vy = jnp.sign(gamma) * F_z * (p["tire.p_vy1"] + p["tire.p_vy3"] * jnp.fabs(gamma))

    alpha_y = alpha + S_hy
    mu_y = p["tire.p_dy1"] * (1 - p["tire.p_dy3"] * gamma**2)

    C_y = p["tire.p_cy1"]
    D_y = mu_y * F_z
    E_y = p["tire.p_ey1"]
    K_y = F_z * p["tire.p_ky1"]  # simplify K_y0 to p.p_ky1*F_z
    B_y = K_y / (C_y * D_y)

    # magic tire formula
    F_y = (
        D_y
        * jnp.sin(
            C_y
            * jnp.arctan(
                B_y * alpha_y - E_y * (B_y * alpha_y - jnp.arctan(B_y * alpha_y))
            )
        )
        + S_vy
    )

    res = []
    res.append(F_y)
    res.append(mu_y)
    return res


# longitudinal tire forces for combined slip
@njit(cache=True)
def formula_longitudinal_comb(kappa, alpha, F0_x, p):
    # turn slip is neglected, so xi_i=1
    # all scaling factors lambda = 1

    S_hxalpha = p["tire.r_hx1"]

    alpha_s = alpha + S_hxalpha

    B_xalpha = p["tire.r_bx1"] * jnp.cos(jnp.arctan(p["tire.r_bx2"] * kappa))
    C_xalpha = p["tire.r_cx1"]
    E_xalpha = p["tire.r_ex1"]
    D_xalpha = F0_x / (
        jnp.cos(
            C_xalpha
            * jnp.arctan(
                B_xalpha * S_hxalpha
                - E_xalpha * (B_xalpha * S_hxalpha - jnp.arctan(B_xalpha * S_hxalpha))
            )
        )
    )

    # magic tire formula
    return D_xalpha * jnp.cos(
        C_xalpha
        * jnp.arctan(
            B_xalpha * alpha_s
            - E_xalpha * (B_xalpha * alpha_s - jnp.arctan(B_xalpha * alpha_s))
        )
    )


# lateral tire forces for combined slip
@njit(cache=True)
def formula_lateral_comb(kappa, alpha, gamma, mu_y, F_z, F0_y, p):
    # turn slip is neglected, so xi_i=1
    # all scaling factors lambda = 1

    S_hykappa = p["tire.r_hy1"]

    kappa_s = kappa + S_hykappa

    B_ykappa = p["tire.r_by1"] * jnp.cos(
        jnp.arctan(p["tire.r_by2"] * (alpha - p["tire.r_by3"]))
    )
    C_ykappa = p["tire.r_cy1"]
    E_ykappa = p["tire.r_ey1"]
    D_ykappa = F0_y / (
        jnp.cos(
            C_ykappa
            * jnp.arctan(
                B_ykappa * S_hykappa
                - E_ykappa * (B_ykappa * S_hykappa - jnp.arctan(B_ykappa * S_hykappa))
            )
        )
    )

    D_vykappa = (
        mu_y
        * F_z
        * (p["tire.r_vy1"] + p["tire.r_vy3"] * gamma)
        * jnp.cos(jnp.arctan(p["tire.r_vy4"] * alpha))
    )
    S_vykappa = D_vykappa * jnp.sin(
        p["tire.r_vy5"] * jnp.arctan(p["tire.r_vy6"] * kappa)
    )

    # magic tire formula
    return (
        D_ykappa
        * jnp.cos(
            C_ykappa
            * jnp.arctan(
                B_ykappa * kappa_s
                - E_ykappa * (B_ykappa * kappa_s - jnp.arctan(B_ykappa * kappa_s))
            )
        )
        + S_vykappa
    )
