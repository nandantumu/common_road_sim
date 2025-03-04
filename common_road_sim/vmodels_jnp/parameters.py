from vehiclemodels.parameters_vehicle1 import parameters_vehicle1 as rv1
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2 as rv2
from vehiclemodels.parameters_vehicle3 import parameters_vehicle3 as rv3
from dataclasses import dataclass, field, fields
from typing import Optional
from numba.typed import Dict
from numba import types


@dataclass
class LongitudinalParameters:
    """
    Class defines all parameters related to the longitudinal dynamics
    """

    # constraints regarding longitudinal dynamics
    v_min: Optional[float] = None  # minimum velocity [m/s]
    v_max: Optional[float] = None  # minimum velocity [m/s]
    v_switch: Optional[float] = None  # switching velocity [m/s]
    a_max: Optional[float] = None  # maximum absolute acceleration [m/s^2]
    j_max: Optional[float] = None  # maximum longitudinal jerk [m/s^3]
    j_dot_max: Optional[float] = None  # maximum change of longitudinal jerk [m/s^4]


@dataclass
class SteeringParameters:
    """
    Class defines all steering related parameters
    """

    # constraints regarding steering
    min: Optional[float] = None  # minimum steering angle [rad]
    max: Optional[float] = None  # maximum steering angle [rad]
    v_min: Optional[float] = None  # minimum steering velocity [rad/s]
    v_max: Optional[float] = None  # maximum steering velocity [rad/s]
    kappa_dot_max: Optional[float] = None  # maximum curvature rate
    kappa_dot_dot_max: Optional[float] = None  # maximum curvature rate rate


@dataclass
class TireParameters:
    """
    Class defines all Tire Parameters
    """

    # tire parameters from ADAMS handbook
    # longitudinal coefficients
    p_cx1: Optional[float] = None  # Shape factor Cfx for longitudinal force
    p_dx1: Optional[float] = None  # Longitudinal friction Mux at Fznom
    p_dx3: Optional[float] = None  # Variation of friction Mux with camber
    p_ex1: Optional[float] = None  # Longitudinal curvature Efx at Fznom
    p_kx1: Optional[float] = None  # Longitudinal slip stiffness Kfx/Fz at Fznom
    p_hx1: Optional[float] = None  # Horizontal shift Shx at Fznom
    p_vx1: Optional[float] = None  # Vertical shift Svx/Fz at Fznom
    r_bx1: Optional[float] = None  # Slope factor for combined slip Fx reduction
    r_bx2: Optional[float] = None  # Variation of slope Fx reduction with kappa
    r_cx1: Optional[float] = None  # Shape factor for combined slip Fx reduction
    r_ex1: Optional[float] = None  # Curvature factor of combined Fx
    r_hx1: Optional[float] = None  # Shift factor for combined slip Fx reduction

    # lateral coefficients
    p_cy1: Optional[float] = None  # Shape factor Cfy for lateral forces
    p_dy1: Optional[float] = None  # Lateral friction Muy
    p_dy3: Optional[float] = None  # Variation of friction Muy with squared camber
    p_ey1: Optional[float] = None  # Lateral curvature Efy at Fznom
    p_ky1: Optional[float] = None  # Maximum value of stiffness Kfy/Fznom
    p_hy1: Optional[float] = None  # Horizontal shift Shy at Fznom
    p_hy3: Optional[float] = None  # Variation of shift Shy with camber
    p_vy1: Optional[float] = None  # Vertical shift in Svy/Fz at Fznom
    p_vy3: Optional[float] = None  # Variation of shift Svy/Fz with camber
    r_by1: Optional[float] = None  # Slope factor for combined Fy reduction
    r_by2: Optional[float] = None  # Variation of slope Fy reduction with alpha
    r_by3: Optional[float] = None  # Shift term for alpha in slope Fy reduction
    r_cy1: Optional[float] = None  # Shape factor for combined Fy reduction
    r_ey1: Optional[float] = None  # Curvature factor of combined Fy
    r_hy1: Optional[float] = None  # Shift factor for combined Fy reduction
    r_vy1: Optional[float] = None  # Kappa induced side force Svyk/Muy*Fz at Fznom
    r_vy3: Optional[float] = None  # Variation of Svyk/Muy*Fz with camber
    r_vy4: Optional[float] = None  # Variation of Svyk/Muy*Fz with alpha
    r_vy5: Optional[float] = None  # Variation of Svyk/Muy*Fz with kappa
    r_vy6: Optional[float] = None  # Variation of Svyk/Muy*Fz with atan(kappa)


@dataclass
class TrailerParameters:
    """
    Class defines all trailer parameters (for on-axle trailer-truck models)
    """

    # class for trailer parameters
    l: Optional[float] = None  # trailer length [m]
    w: Optional[float] = None  # trailer width [m]
    l_hitch: Optional[float] = None  # hitch length [m]
    l_total: Optional[float] = None  # total system length [m]
    l_wb: Optional[float] = None  # trailer wheel base [m]


@dataclass
class VehicleParameters:
    """
    VehicleParameters base class: defines all parameters used by the vehicle models described in
    Althoff, M. and Würsching, G. "CommonRoad: Vehicle Models", 2020
    """

    # vehicle body dimensions
    l: Optional[float] = None
    w: Optional[float] = None

    # steering parameters
    steering: SteeringParameters = field(default_factory=SteeringParameters)

    # longitudinal parameters
    longitudinal: LongitudinalParameters = field(default_factory=LongitudinalParameters)

    # masses
    m: Optional[float] = None
    m_s: Optional[float] = None
    m_uf: Optional[float] = None
    m_ur: Optional[float] = None

    # axes distances
    a: Optional[float] = (
        None  # distance from spring mass center of gravity to front axle [m]  LENA
    )
    b: Optional[float] = (
        None  # distance from spring mass center of gravity to rear axle [m]  LENB
    )

    # moments of inertia of sprung mass
    I_Phi_s: Optional[float] = (
        None  # moment of inertia for sprung mass in roll [kg m^2]  IXS
    )
    I_y_s: Optional[float] = (
        None  # moment of inertia for sprung mass in pitch [kg m^2]  IYS
    )
    I_z: Optional[float] = (
        None  # moment of inertia for sprung mass in yaw [kg m^2]  IZZ
    )
    I_xz_s: Optional[float] = None  # moment of inertia cross product [kg m^2]  IXZ

    # suspension parameters
    K_sf: Optional[float] = None  # suspension spring rate (front) [N/m]  KSF
    K_sdf: Optional[float] = None  # suspension damping rate (front) [N s/m]  KSDF
    K_sr: Optional[float] = None  # suspension spring rate (rear) [N/m]  KSR
    K_sdr: Optional[float] = None  # suspension damping rate (rear) [N s/m]  KSDR

    # geometric parameters
    T_f: Optional[float] = None  # track width front [m]  TRWF
    T_r: Optional[float] = None  # track width rear [m]  TRWB
    K_ras: Optional[float] = (
        None  # lateral spring rate at compliant compliant pin joint between M_s and M_u [N/m]  KRAS
    )

    K_tsf: Optional[float] = (
        None  # auxiliary torsion roll stiffness per axle (normally negative) (front) [N m/rad]  KTSF
    )
    K_tsr: Optional[float] = (
        None  # auxiliary torsion roll stiffness per axle (normally negative) (rear) [N m/rad]  KTSR
    )
    K_rad: Optional[float] = (
        None  # damping rate at compliant compliant pin joint between M_s and M_u [N s/m]  KRADP
    )
    K_zt: Optional[float] = None  # vertical spring rate of tire [N/m]  TSPRINGR

    h_cg: Optional[float] = (
        None  # center of gravity height of total mass [m]  HCG (mainly required for conversion to other vehicle models)
    )
    h_raf: Optional[float] = None  # height of roll axis above ground (front) [m]  HRAF
    h_rar: Optional[float] = None  # height of roll axis above ground (rear) [m]  HRAR

    h_s: Optional[float] = None  # M_s center of gravity above ground [m]  HS

    I_uf: Optional[float] = (
        None  # moment of inertia for unsprung mass about x-axis (front) [kg m^2]  IXUF
    )
    I_ur: Optional[float] = (
        None  # moment of inertia for unsprung mass about x-axis (rear) [kg m^2]  IXUR
    )
    I_y_w: Optional[float] = (
        None  # wheel inertia, from internet forum for 235/65 R 17 [kg m^2]
    )

    K_lt: Optional[float] = (
        None  # lateral compliance rate of tire, wheel, and suspension, per tire [m/N]  KLT
    )
    R_w: Optional[float] = (
        None  # effective wheel/tire radius  chosen as tire rolling radius RR  taken from ADAMS documentation [m]
    )

    # split of brake and engine torque
    T_sb: Optional[float] = None
    T_se: Optional[float] = None

    # suspension parameters
    D_f: Optional[float] = None  # [rad/m]  DF
    D_r: Optional[float] = None  # [rad/m]  DR
    E_f: Optional[float] = None  # [needs conversion if nonzero]  EF
    E_r: Optional[float] = None  # [needs conversion if nonzero]  ER

    # tire parameters
    tire: TireParameters = field(default_factory=TireParameters)

    # trailer parameters
    trailer: TrailerParameters = field(default_factory=TrailerParameters)


def convert_omegaconf_to_dataclass(omegaconf_dict):
    """
    Convert OmegaConf dictionary to dataclass
    """
    vp = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    for vehicle_property in fields(VehicleParameters):
        if vehicle_property.name not in ["steering", "longitudinal", "tire", "trailer"]:
            item_value = getattr(omegaconf_dict, vehicle_property.name)
            if item_value is not None:
                vp[vehicle_property.name] = item_value
        elif vehicle_property.name == "steering":
            for subfield in fields(SteeringParameters):
                item_value = getattr(omegaconf_dict.steering, subfield.name)
                if item_value is not None:
                    vp[vehicle_property.name + "." + subfield.name] = getattr(
                        omegaconf_dict.steering, subfield.name
                    )
        elif vehicle_property.name == "longitudinal":
            for subfield in fields(LongitudinalParameters):
                item_value = getattr(omegaconf_dict.longitudinal, subfield.name)
                if item_value is not None:
                    vp[vehicle_property.name + "." + subfield.name] = getattr(
                        omegaconf_dict.longitudinal, subfield.name
                    )
        elif vehicle_property.name == "tire":
            for subfield in fields(TireParameters):
                item_value = getattr(omegaconf_dict.tire, subfield.name)
                if item_value is not None:
                    vp[vehicle_property.name + "." + subfield.name] = getattr(
                        omegaconf_dict.tire, subfield.name
                    )
        elif vehicle_property.name == "trailer":
            for subfield in fields(TrailerParameters):
                item_value = getattr(omegaconf_dict.trailer, subfield.name)
                if item_value is not None:
                    vp[vehicle_property.name + "." + subfield.name] = getattr(
                        omegaconf_dict.trailer, subfield.name
                    )
    return vp


def parameters_vehicle1():
    return convert_omegaconf_to_dataclass(rv1())


def parameters_vehicle2():
    return convert_omegaconf_to_dataclass(rv2())


def parameters_vehicle3():
    return convert_omegaconf_to_dataclass(rv3())
