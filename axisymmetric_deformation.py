import os
import gc
import numpy as np
from mayavi import mlab
from colour import Color
from PyQt5 import QtWidgets, QtCore, QtGui
from numbers import Number
from typing import Dict, Callable, Tuple
from scipy.integrate import solve_ivp

# Spiral angle method and coefficients definition:
# Constant spiral angle:
# ALPHA_METHOD = 'constant'
# ALPHA_METHOD = 'poly'
COEFFS = [0]
POLYNOMIAL_ORDER = 0
# SPHERICAL CAP
ALPHA_METHOD = 'sphere'
SPHERE_RAD = 190 * 1e-3  # m
GAUSSIAN_CURVATURE = 1/(SPHERE_RAD**2)
# GAUSSIAN_CURVATURE = 1

# Fluid parameters
VISCOSITY = 60  # [Pa * s]
CRITICAL_PRESSURE = 262 * 1e3  # [Pa]
# INLET_PRESSURE = 900 * 1e3  Inlet pressure [Pa]
INLET_PRESSURE = 600 * 1e3  # Inlet pressure [Pa]

# Geometric parameters
NUM_FRUSTA = 78  # Number of frusta
MIN_LENGTH = 80 * 1e-3  # m
MAX_LENGTH = 415 * 1e-3  # m
FRUSTUM_STATIC_LEN = (MAX_LENGTH + MIN_LENGTH) / (2*NUM_FRUSTA)
# Single frustum dynamic part length, h [m]:
DYN_FRUSTUM_LEN = (MAX_LENGTH - MIN_LENGTH) / (2*NUM_FRUSTA)
# Single frustum length, l_open - total [m]:
PROTRUDED_LEN = FRUSTUM_STATIC_LEN + DYN_FRUSTUM_LEN
INNER_RAD = 12 * 1e-3  # Straw inner radius [m]
RAD_MIN = 55 * 1e-3  # Structure smallest radius [m]
RAD_MAX = RAD_MIN + NUM_FRUSTA*(PROTRUDED_LEN-2*DYN_FRUSTUM_LEN)  # Structure largest radius [m]
THETA_MIN = 0
THETA_MAX = 2*np.pi
# Pressure-Deformation graph unstable slope, k2
SNAP_SLOPE = CRITICAL_PRESSURE / DYN_FRUSTUM_LEN  # [Pa/m]

# Non-dimensional parameters:
# non-dimensional deformation of a single frustum, Pi_L
DEFORMATION_NORM = 2 * DYN_FRUSTUM_LEN / PROTRUDED_LEN
# Non-dimensional pressure difference, Pi_K
PRESSURE_DIFF_NORM = ((INLET_PRESSURE - CRITICAL_PRESSURE - SNAP_SLOPE*DYN_FRUSTUM_LEN)
                      / (SNAP_SLOPE * PROTRUDED_LEN))

# Stretch along director, lambda
PROTRUDED_DEFORMATION = (PROTRUDED_LEN
                         / (PROTRUDED_LEN - 2*DYN_FRUSTUM_LEN))
# TRANSITION_DEFORMATION_METHOD = 'exponential'
TRANSITION_DEFORMATION_METHOD = 'poly'
TRANSITION_LEN = 20 * PROTRUDED_LEN
# Number of sampling points
NUM_RAD_SAMPLES = 200
# NUM_RAD_SAMPLES = 1000
NUM_THETA_SAMPLES = 50

DEF_RTOL = 1e-8
DEF_ATOL = 1e-8

END_TIME_NORM = ((NUM_FRUSTA + 1) * NUM_FRUSTA/2
                 * np.log(DEFORMATION_NORM / PRESSURE_DIFF_NORM))
END_TIME = (8 * VISCOSITY * PROTRUDED_LEN
            / (SNAP_SLOPE * INNER_RAD**2)
            * END_TIME_NORM)
RAD_MIN_THRESHOLD = 1e-8


# Director angle function
def polynomial_angle(radius: np.ndarray | Number,
                     coeffs: np.ndarray):
    """Compute angle and angle derivative using a polynomial function of the radius.

    Args:
        radius: Radius values.
        coeffs: Coefficient vector.

    Returns:
        Angle and angle derivatives with respect to the radius.
    """
    radius = np.array(radius) if type(radius) is not np.ndarray else radius
    # Compute polynomial function
    angle_poly = np.polynomial.Polynomial(coeffs)
    return angle_poly(radius), angle_poly.deriv(1)(radius)


def sphere_deformation_angle(radius: np.ndarray | Number,
                             gaussian_curvature: Number,
                             protruded_deformation: Number):
    """Compute angle and angle derivative that result in deformation into a
    sphere.

    Args:
        radius: Radius values.
        gaussian_curvature: Desired Gaussian Curvature value.
        deformation: Deformation of the director field.

    Returns:
        Angle and angle derivatives with respect to the radius.
    """
    radius = np.array(radius) if type(radius) is not np.ndarray else radius
    # Compute constants
    deformation_const = 1 - 2 / (1 + protruded_deformation)  # c
    curvature_const = 1/2 * gaussian_curvature / (1 - protruded_deformation**(-2))  # C(K)
    # Compute expression inside arccos in spiral
    cosine_expr = curvature_const * radius**2 + deformation_const
    # Find maximum radius for which the cosine expression equals to 1
    max_rad = np.sqrt((1 - deformation_const) / (curvature_const))
    # Initialize angle and angle derivative
    angle = np.zeros(radius.shape)
    d_cosine_expr = np.zeros(radius.shape)
    d_angle = np.zeros(angle.shape)
    # Compute angle.
    # The angle is defined as 0 for all radii larger than the
    # maximum radius (will result in a cone)
    angle[radius < max_rad] = np.arccos(cosine_expr[radius < max_rad]) / 2
    # Compute angle derivative
    d_cosine_expr[radius < max_rad] = 2 * curvature_const * radius[radius < max_rad]
    d_angle[radius < max_rad] = (-d_cosine_expr[radius < max_rad]
                                 / (np.sqrt(1 - cosine_expr[radius < max_rad]**2))
                                 / 2)
    # Constant (non 0) radius for which the circumference does not change after deformation.
    # This radius does not exist for a sphere, as the deformation_const < 1 by definition.
    # If the deformation in the normal direction to the directors would not have been equal to 1,
    # this radius may have existed.
    # constant_radius = (np.sqrt(2) / (protruded_deformation * np.sqrt(gaussian_curvature))
    #                    * np.sqrt((deformation_const - 1) * (1 + protruded_deformation**2)))
    # print(f'{constant_radius = }')
    return angle, d_angle


# %ds functions
def arc_len_diff(angle: np.ndarray | Number):
    """Compute the integral curve arc-length derivatives with respect to the radius.

    Args:
        angle: Integral curve tangent angles with respect to the x axis.

    Returns:
        Integral curve arc-length derivatives
    """
    return 1 / np.cos(angle)


# %% Calculation functions
def compute_shape(
        director_angle_func: Callable[..., Tuple[np.ndarray, np.ndarray]],
        angle_func_args=None,
        end_time: Number = END_TIME,
        num_timesteps: int = 100,
        transition_len: Number = TRANSITION_LEN,
        protruded_len: Number = PROTRUDED_LEN,
        dyn_frustum_len: Number = DYN_FRUSTUM_LEN,
        snap_slope: Number = SNAP_SLOPE,
        viscosity: Number = VISCOSITY,
        inner_rad: Number = INNER_RAD,
        inlet_pressure: Number = INLET_PRESSURE,
        critical_pressure: Number = CRITICAL_PRESSURE,
        protruded_deformation: Number = PROTRUDED_DEFORMATION,
        rad_min: Number = RAD_MIN,
        rad_max: Number = RAD_MAX,
        theta_min: Number = THETA_MIN,
        theta_max: Number = THETA_MAX,
        num_rad_samples: int = NUM_RAD_SAMPLES,
        num_theta_samples: int = NUM_THETA_SAMPLES,
        rtol: Number = DEF_RTOL,
        atol: Number = DEF_ATOL,
        ) -> np.ndarray:
    timesteps = np.linspace(0, end_time, num_timesteps)
    data = compute_deformed_surface(
        timesteps, director_angle_func, angle_func_args, transition_len,
        protruded_len, dyn_frustum_len, snap_slope, viscosity, inner_rad,
        inlet_pressure, critical_pressure, protruded_deformation,
        rad_min, rad_max, theta_min, theta_max,
        num_rad_samples, num_theta_samples,
        rtol, atol)
    return dict(timesteps=timesteps, **data)


def compute_deformed_surface(
        timesteps: np.ndarray,
        director_angle_func: Callable[..., Tuple[np.ndarray, np.ndarray]],
        angle_func_args=None,
        transition_len: Number = TRANSITION_LEN,
        protruded_len: Number = PROTRUDED_LEN,
        dyn_frustum_len: Number = DYN_FRUSTUM_LEN,
        snap_slope: Number = SNAP_SLOPE,
        viscosity: Number = VISCOSITY,
        inner_rad: Number = INNER_RAD,
        inlet_pressure: Number = INLET_PRESSURE,
        critical_pressure: Number = CRITICAL_PRESSURE,
        protruded_deformation: Number = PROTRUDED_DEFORMATION,
        rad_min: Number = RAD_MIN,
        rad_max: Number = RAD_MAX,
        theta_min: Number = THETA_MIN,
        theta_max: Number = THETA_MAX,
        num_rad_samples: int = NUM_RAD_SAMPLES,
        num_theta_samples: int = NUM_THETA_SAMPLES,
        rtol: Number = DEF_RTOL,
        atol: Number = DEF_ATOL,
        ) -> np.ndarray:
    radius = np.linspace(rad_min, rad_max, num_rad_samples)
    theta = np.linspace(theta_min, theta_max, num_theta_samples)
    # The shape of the deformed curves is (num_timesteps, num_rad_samples):
    integration_data = compute_deformed_cross_section_curve(
        radius, timesteps, director_angle_func, angle_func_args,
        transition_len, protruded_len, dyn_frustum_len,
        snap_slope, viscosity, inner_rad, inlet_pressure,
        critical_pressure, protruded_deformation,
        rad_min, rtol, atol)  # g2

    deformed_curve = integration_data['deformed_curve']
    deformed_cross_section_curve_x = deformed_curve['deformed_cross_section_curve_x']
    deformed_cross_section_curve_z = deformed_curve['deformed_cross_section_curve_z']

    # The shape of the deformed surface coords is
    # (num_timesteps, num_rad_samples, num_theta_samples):
    deformed_surf_x = (deformed_cross_section_curve_x[:, :, np.newaxis]
                       * np.cos(theta).reshape(1, 1, -1))
    deformed_surf_y = (deformed_cross_section_curve_x[:, :, np.newaxis]
                       * np.sin(theta).reshape(1, 1, -1))
    deformed_surf_z = np.tile(deformed_cross_section_curve_z[:, :, np.newaxis],
                              (1, 1, len(theta)))

    return dict(
        # The overall returned shape of the surface
        # (num_timesteps, num_coords(=3), num_rad_samples, num_theta_samples)
        surface=np.stack([deformed_surf_x, deformed_surf_y, deformed_surf_z], axis=1),
        # Deformed cross section curve: (num_timesteps, num_rad_samples)
        deformed_cross_section_curve_x=deformed_cross_section_curve_x,
        deformed_cross_section_curve_z=deformed_cross_section_curve_z,
        # Deformed integral curve theta values
        deformed_director_theta=integration_data['deformed_director_theta'],
        deformed_normal_theta=integration_data['deformed_normal_theta'],
        # (1, num_rad_samples)
        director_arc_len=integration_data['director_arc_len'],
        director_theta=integration_data['director_theta'],
        normal_curve_theta=integration_data['normal_curve_theta'],
        # (num_timesteps, num_rad_samples)
        rotational_deformation_angle=integration_data['rotational_deformation_angle']
        )


def compute_deformed_cross_section_curve(
        radius: np.ndarray,
        timesteps: np.ndarray,
        director_angle_func: Callable[..., Tuple[np.ndarray, np.ndarray]],
        angle_func_args=None,
        transition_len: Number = TRANSITION_LEN,
        protruded_len: Number = PROTRUDED_LEN,
        dyn_frustum_len: Number = DYN_FRUSTUM_LEN,
        snap_slope: Number = SNAP_SLOPE,
        viscosity: Number = VISCOSITY,
        inner_rad: Number = INNER_RAD,
        inlet_pressure: Number = INLET_PRESSURE,
        critical_pressure: Number = CRITICAL_PRESSURE,
        protruded_deformation: Number = PROTRUDED_DEFORMATION,
        rad_min: Number = RAD_MIN,
        rtol: Number = DEF_RTOL,
        atol: Number = DEF_ATOL,
        ) -> np.ndarray:
    angle_func_args = [] if angle_func_args is None else angle_func_args
    # r_span = (rad_min, radius.max())
    r_span = (RAD_MIN_THRESHOLD, radius.max())
    r_eval = radius
    inner_radius_z_coord = 0
    init_director_arc_len = 0
    init_director_theta = 0
    integration_init = {
        'z_coords': np.ones(len(timesteps)) * inner_radius_z_coord,
        'deformed_director_theta': np.ones(len(timesteps)) * init_director_theta,
        'deformed_normal_theta': np.ones(len(timesteps)) * init_director_theta,
        'director_arc_len': np.array([init_director_arc_len]),
        'director_theta': np.array([init_director_theta]),
        'normal_curve_theta': np.array([init_director_theta]),
        'deformed_radius': np.ones(len(timesteps)) * np.array([RAD_MIN_THRESHOLD]),
    }
    idx_cumsum = np.cumsum([len(val) for val in integration_init.values()])
    start_idx = dict(zip(integration_init.keys(), np.concatenate([[0], idx_cumsum[:-1]])))
    end_idx = dict(zip(integration_init.keys(), idx_cumsum))
    y0 = np.concatenate([val for val in integration_init.values()])
    sol = solve_ivp(compute_d_deformed_cross_section_curve_data, r_span,
                    y0=y0,
                    t_eval=r_eval,
                    args=(
                        timesteps, start_idx, end_idx, director_angle_func,
                        angle_func_args, transition_len, protruded_len,
                        dyn_frustum_len, snap_slope, viscosity, inner_rad,
                        inlet_pressure, critical_pressure, protruded_deformation
                        ),
                    rtol=rtol, atol=atol,
                    vectorized=True)

    deformed_cross_section_curve_z = sol.y[start_idx['z_coords']: end_idx['z_coords']]
    director_arc_len = sol.y[start_idx['director_arc_len']: end_idx['director_arc_len']]
    director_theta = sol.y[start_idx['director_theta']: end_idx['director_theta']]
    normal_curve_theta = sol.y[start_idx['normal_curve_theta']: end_idx['normal_curve_theta']]
    deformed_director_theta = sol.y[start_idx['deformed_director_theta']: end_idx['deformed_director_theta']]
    deformed_normal_theta = sol.y[start_idx['deformed_normal_theta']: end_idx['deformed_normal_theta']]

    deformed_cross_section_curve_x, _ = compute_deformed_cross_section_curve_x(
        radius, director_arc_len.reshape(1, -1), timesteps,
        director_angle_func, angle_func_args,
        transition_len, protruded_len, dyn_frustum_len,
        snap_slope, viscosity, inner_rad, inlet_pressure,
        critical_pressure, protruded_deformation)

    rotational_deformation_angle = compute_rotational_deformation_angle(
        radius, director_arc_len.reshape(1, -1), timesteps,
        director_angle_func, angle_func_args,
        transition_len, protruded_len, dyn_frustum_len,
        snap_slope, viscosity, inner_rad, inlet_pressure,
        critical_pressure, protruded_deformation)  # gamma

    deformed_curve = {
        'deformed_cross_section_curve_x': deformed_cross_section_curve_x,
        'deformed_cross_section_curve_z': deformed_cross_section_curve_z,
    }

    return dict(deformed_curve=deformed_curve,
                director_arc_len=director_arc_len,
                director_theta=director_theta,
                normal_curve_theta=normal_curve_theta,
                rotational_deformation_angle=rotational_deformation_angle,
                deformed_director_theta=deformed_director_theta,
                deformed_normal_theta=deformed_normal_theta)


def compute_d_deformed_cross_section_curve_data(
        radius: np.ndarray,
        deformed_curve_data: np.ndarray,
        timesteps: np.ndarray,
        start_idx: dict,
        end_idx: dict,
        director_angle_func: Callable[..., Tuple[np.ndarray, np.ndarray]],
        angle_func_args=None,
        transition_len: Number = TRANSITION_LEN,
        protruded_len: Number = PROTRUDED_LEN,
        dyn_frustum_len: Number = DYN_FRUSTUM_LEN,
        snap_slope: Number = SNAP_SLOPE,
        viscosity: Number = VISCOSITY,
        inner_rad: Number = INNER_RAD,
        inlet_pressure: Number = INLET_PRESSURE,
        critical_pressure: Number = CRITICAL_PRESSURE,
        protruded_deformation: Number = PROTRUDED_DEFORMATION):
    """Compute the derivatives of the deformed z-components
    for all timesteps, and the derivative of the un-deformed director
    integral curve.

    Args:
        radius:
            Radius values.
        deformed_curve_data:
            Data of the current deformed z-component and the director
            arc length
        timesteps:
            Array of timesteps for computation.
        director_angle_func:
            Function that computes of the director angle at every radius
        angle_func_args:
            Additional arguments needed for director_angle_func. Defaults to None.

    Returns:
        Array of derivatives at the given radius values.
    """
    director_arc_len = deformed_curve_data[
        start_idx['director_arc_len']: end_idx['director_arc_len']]

    deformed_radius = deformed_curve_data[
        start_idx['deformed_radius']: end_idx['deformed_radius']]

    if len(deformed_curve_data.shape) < 1:
        director_arc_len = director_arc_len[:, np.newaxis]

    d_deformed_curve_z = compute_d_deformed_cross_section_curve_z(
        radius, director_arc_len, timesteps,
        director_angle_func, angle_func_args,
        transition_len, protruded_len, dyn_frustum_len,
        snap_slope, viscosity, inner_rad, inlet_pressure,
        critical_pressure, protruded_deformation)

    d_director_arc_len = (compute_d_director_arc_len(
        radius, director_angle_func, angle_func_args)
                          .reshape(1, -1))

    d_deformed_radius = compute_d_deformed_radius(
        radius, director_arc_len, timesteps, director_angle_func,
        angle_func_args, transition_len, protruded_len,
        dyn_frustum_len, snap_slope, viscosity, inner_rad,
        inlet_pressure, critical_pressure, protruded_deformation)

    d_deformed_integral_curve_theta = compute_d_deformed_integral_curves_theta(
        radius, director_arc_len, deformed_radius, timesteps,
        director_angle_func, angle_func_args,
        transition_len, protruded_len, dyn_frustum_len,
        snap_slope, viscosity, inner_rad, inlet_pressure,
        critical_pressure, protruded_deformation)

    angle_func_args = [] if angle_func_args is None else angle_func_args
    director_angle, _ = director_angle_func(radius, *angle_func_args)
    d_director_theta = (np.tan(director_angle) / radius
                        if radius >= RAD_MIN_THRESHOLD else np.array([0])).reshape(1, -1)
    d_normal_curve_theta = (np.tan(director_angle - np.pi/2) / radius
                            if radius >= RAD_MIN_THRESHOLD else np.array([0])).reshape(1, -1)

    # Create a vector of derivative for integration
    d_deformed_curve_data = np.zeros_like(deformed_curve_data)
    d_deformed_curve_data[
        start_idx['z_coords']: end_idx['z_coords']] = d_deformed_curve_z
    d_deformed_curve_data[
        start_idx['deformed_director_theta']:
            end_idx['deformed_director_theta']] = (
                d_deformed_integral_curve_theta['d_deformed_director_theta'])
    d_deformed_curve_data[
        start_idx['deformed_normal_theta']:
            end_idx['deformed_normal_theta']] = (
                d_deformed_integral_curve_theta['d_deformed_normal_theta'])
    d_deformed_curve_data[
        start_idx['director_arc_len']:
            end_idx['director_arc_len']] = d_director_arc_len
    d_deformed_curve_data[
        start_idx['director_theta']:
            end_idx['director_theta']] = d_director_theta
    d_deformed_curve_data[
        start_idx['normal_curve_theta']:
            end_idx['normal_curve_theta']] = d_normal_curve_theta
    d_deformed_curve_data[
        start_idx['deformed_radius']:
            end_idx['deformed_radius']] = d_deformed_radius

    return d_deformed_curve_data


def compute_d_deformed_cross_section_curve_z(
        radius: np.ndarray,
        director_arc_len: np.ndarray,
        timesteps: np.ndarray,
        director_angle_func: Callable[..., Tuple[np.ndarray, np.ndarray]],
        angle_func_args=None,
        transition_len: Number = TRANSITION_LEN,
        protruded_len: Number = PROTRUDED_LEN,
        dyn_frustum_len: Number = DYN_FRUSTUM_LEN,
        snap_slope: Number = SNAP_SLOPE,
        viscosity: Number = VISCOSITY,
        inner_rad: Number = INNER_RAD,
        inlet_pressure: Number = INLET_PRESSURE,
        critical_pressure: Number = CRITICAL_PRESSURE,
        protruded_deformation: Number = PROTRUDED_DEFORMATION
        ) -> np.ndarray:
    func_input = [
        radius, director_arc_len, timesteps,
        director_angle_func, angle_func_args,
        transition_len, protruded_len, dyn_frustum_len,
        snap_slope, viscosity, inner_rad, inlet_pressure,
        critical_pressure, protruded_deformation
        ]
    d_deformed_radius = compute_d_deformed_radius(*func_input)
    _, d_deformed_curve_x = compute_deformed_cross_section_curve_x(*func_input)
    d_deformed_curve_z_squared = d_deformed_radius**2 - d_deformed_curve_x**2
    d_deformed_curve_z_squared[d_deformed_curve_z_squared < 0] = 0

    return np.sqrt(d_deformed_curve_z_squared)


def compute_d_deformed_integral_curves_theta(
        radius: np.ndarray,
        director_arc_len: np.ndarray,
        deformed_radius: np.ndarray,
        timesteps: np.ndarray,
        director_angle_func: Callable[..., Tuple[np.ndarray, np.ndarray]],
        angle_func_args=None,
        transition_len: Number = TRANSITION_LEN,
        protruded_len: Number = PROTRUDED_LEN,
        dyn_frustum_len: Number = DYN_FRUSTUM_LEN,
        snap_slope: Number = SNAP_SLOPE,
        viscosity: Number = VISCOSITY,
        inner_rad: Number = INNER_RAD,
        inlet_pressure: Number = INLET_PRESSURE,
        critical_pressure: Number = CRITICAL_PRESSURE,
        protruded_deformation: Number = PROTRUDED_DEFORMATION
        ) -> Dict[str, np.ndarray]:
    # Compute director angle for the given radius and director angle function
    angle_func_args = [] if angle_func_args is None else angle_func_args
    director_angle, _ = director_angle_func(radius, *angle_func_args)
    # Compute the rotation angle between the deformed director curve to
    # the new radial direction.
    director_rotation_angle = compute_rotational_deformation_angle(
        radius, director_arc_len.reshape(1, -1), timesteps,
        director_angle_func, angle_func_args,
        transition_len, protruded_len, dyn_frustum_len,
        snap_slope, viscosity, inner_rad, inlet_pressure,
        critical_pressure, protruded_deformation)  # gamma
    # Compute the derivative of the deformed radius values
    d_deformed_radius = compute_d_deformed_radius(
        radius, director_arc_len, timesteps, director_angle_func,
        angle_func_args, transition_len, protruded_len,
        dyn_frustum_len, snap_slope, viscosity, inner_rad,
        inlet_pressure, critical_pressure, protruded_deformation)
    # Compute the derivative of the deformed director curve angle
    # on the 3D surface.
    d_deformed_director_theta = np.zeros_like(deformed_radius)
    d_deformed_director_theta[deformed_radius >= RAD_MIN_THRESHOLD] = (
        np.tan(director_angle
               - director_rotation_angle[deformed_radius >= RAD_MIN_THRESHOLD])
        / deformed_radius[deformed_radius >= RAD_MIN_THRESHOLD]
        * d_deformed_radius[deformed_radius >= RAD_MIN_THRESHOLD])
    # Compute the derivative of the deformed normal integral curve angle
    # on the 3D surface.
    d_deformed_normal_theta = np.zeros_like(deformed_radius)
    d_deformed_normal_theta[deformed_radius >= RAD_MIN_THRESHOLD] = (
        np.tan(director_angle
               - director_rotation_angle[deformed_radius >= RAD_MIN_THRESHOLD]
               - np.pi/2)
        / deformed_radius[deformed_radius >= RAD_MIN_THRESHOLD]
        * d_deformed_radius[deformed_radius >= RAD_MIN_THRESHOLD])
    # Return angle derivative values for integral curves
    return {'d_deformed_director_theta': d_deformed_director_theta,
            'd_deformed_normal_theta': d_deformed_normal_theta}


def compute_deformed_cross_section_curve_x(
        radius: np.ndarray,
        director_arc_len: np.ndarray,
        timesteps: np.ndarray,
        director_angle_func: Callable[..., Tuple[np.ndarray, np.ndarray]],
        angle_func_args=None,
        transition_len: Number = TRANSITION_LEN,
        protruded_len: Number = PROTRUDED_LEN,
        dyn_frustum_len: Number = DYN_FRUSTUM_LEN,
        snap_slope: Number = SNAP_SLOPE,
        viscosity: Number = VISCOSITY,
        inner_rad: Number = INNER_RAD,
        inlet_pressure: Number = INLET_PRESSURE,
        critical_pressure: Number = CRITICAL_PRESSURE,
        protruded_deformation: Number = PROTRUDED_DEFORMATION
        ) -> Tuple[np.ndarray, np.ndarray]:
    angle_func_args = [] if angle_func_args is None else angle_func_args
    director_angle, d_director_angle = director_angle_func(radius, *angle_func_args)
    d_director_arc_len = compute_d_director_arc_len(radius, director_angle_func,
                                                    angle_func_args)

    deformation, d_deformation = compute_deformation_per_radius(
        radius, director_arc_len, d_director_arc_len, timesteps,
        transition_len, protruded_len, dyn_frustum_len, snap_slope,
        viscosity, inner_rad, inlet_pressure, critical_pressure,
        protruded_deformation)

    cos_a = np.cos(director_angle)
    sin_a = np.sin(director_angle)

    metric_tensor_rr = cos_a**2 + sin_a**2 * deformation**2
    sqrt_metric_tensor_rr = np.sqrt(metric_tensor_rr)
    deformed_cross_section_curve_x = radius * sqrt_metric_tensor_rr

    # Consider making a function of the derivative using sympy
    # 2*sin(a)*(cos(a)*da(L**2 - 1) + sin_a**L*dL)
    d_metric_tensor_rr = (2*sin_a
                          * (cos_a * d_director_angle * (deformation**2 - 1)
                             + sin_a * deformation * d_deformation))
    d_sqrt_metric_tensor_rr = 1/2 * d_metric_tensor_rr / sqrt_metric_tensor_rr
    d_deformed_curve_x = sqrt_metric_tensor_rr + radius * d_sqrt_metric_tensor_rr

    return deformed_cross_section_curve_x, d_deformed_curve_x


def compute_rotational_deformation_angle(
        radius: np.ndarray,
        director_arc_len: np.ndarray,
        timesteps: np.ndarray,
        director_angle_func: Callable[..., Tuple[np.ndarray, np.ndarray]],
        angle_func_args=None,
        transition_len: Number = TRANSITION_LEN,
        protruded_len: Number = PROTRUDED_LEN,
        dyn_frustum_len: Number = DYN_FRUSTUM_LEN,
        snap_slope: Number = SNAP_SLOPE,
        viscosity: Number = VISCOSITY,
        inner_rad: Number = INNER_RAD,
        inlet_pressure: Number = INLET_PRESSURE,
        critical_pressure: Number = CRITICAL_PRESSURE,
        protruded_deformation: Number = PROTRUDED_DEFORMATION
        ) -> np.ndarray:
    angle_func_args = [] if angle_func_args is None else angle_func_args
    director_angle, _ = director_angle_func(radius, *angle_func_args)
    d_director_arc_len = compute_d_director_arc_len(radius, director_angle_func,
                                                    angle_func_args)

    deformation, _ = compute_deformation_per_radius(
        radius, director_arc_len, d_director_arc_len, timesteps,
        transition_len, protruded_len, dyn_frustum_len, snap_slope,
        viscosity, inner_rad, inlet_pressure, critical_pressure,
        protruded_deformation)

    rotational_deformation_angle = np.arctan2(
        np.tan(director_angle) * (1 - deformation),
        1 + deformation*(np.tan(director_angle)**2)
        )
    # rotational_deformation_angle = np.arctan2(
    #     np.tan(director_angle) * (1 - deformation),
    #     1 + (np.tan(director_angle)**2)
    #     )
    # rotational_deformation_angle = np.arctan(
    #     (np.tan(director_angle) * (1 - deformation))
    #     / (1 + deformation*(np.tan(director_angle)**2))
    #     )

    return rotational_deformation_angle


def compute_d_deformed_radius(
        radius: np.ndarray,
        director_arc_len: np.ndarray,
        timesteps: np.ndarray,
        director_angle_func: Callable[..., Tuple[np.ndarray, np.ndarray]],
        angle_func_args=None,
        transition_len: Number = TRANSITION_LEN,
        protruded_len: Number = PROTRUDED_LEN,
        dyn_frustum_len: Number = DYN_FRUSTUM_LEN,
        snap_slope: Number = SNAP_SLOPE,
        viscosity: Number = VISCOSITY,
        inner_rad: Number = INNER_RAD,
        inlet_pressure: Number = INLET_PRESSURE,
        critical_pressure: Number = CRITICAL_PRESSURE,
        protruded_deformation: Number = PROTRUDED_DEFORMATION
        ) -> np.ndarray:  # du
    # This is repeated from compute_deformed_curve_x
    angle_func_args = [] if angle_func_args is None else angle_func_args
    director_angle, _ = director_angle_func(radius, *angle_func_args)
    d_director_arc_len = compute_d_director_arc_len(radius, director_angle_func,
                                                    angle_func_args)
    deformation, _ = compute_deformation_per_radius(
        radius, director_arc_len, d_director_arc_len, timesteps, transition_len,
        protruded_len, dyn_frustum_len, snap_slope, viscosity,
        inner_rad, inlet_pressure, critical_pressure, protruded_deformation)
    cos_a = np.cos(director_angle)
    sin_a = np.sin(director_angle)
    metric_tensor_rr = cos_a**2 + sin_a**2 * deformation**2
    # Up to here
    return deformation / np.sqrt(metric_tensor_rr)


def compute_protruded_arc_len(
        timesteps: np.ndarray,
        protruded_len: Number = PROTRUDED_LEN,
        snap_slope: Number = SNAP_SLOPE,
        dyn_frustum_len: Number = DYN_FRUSTUM_LEN,
        viscosity: Number = VISCOSITY,
        inner_rad: Number = INNER_RAD,
        inlet_pressure: Number = INLET_PRESSURE,
        critical_pressure: Number = CRITICAL_PRESSURE,
        protruded_deformation: Number = PROTRUDED_DEFORMATION
        ) -> np.ndarray:
    # Normalize time
    timesteps_norm = timesteps * (inner_rad**2 * snap_slope)/(8*viscosity*protruded_len)  # (n-1,)
    # Normalize pressure difference (Pi_k)
    pressure_diff_norm = (
        (inlet_pressure - critical_pressure - snap_slope*dyn_frustum_len)
        / (snap_slope * protruded_len))
    # Normalize snapping deformation (Pi_l)
    deformation_norm = 2 * dyn_frustum_len / protruded_len
    # Compute time factor in long wave approximation
    time_factor = (2*deformation_norm
                   / np.log(1 + deformation_norm/pressure_diff_norm))  # beta
    # Normalized protrusion "Wave front" according to long wave approximation
    protruded_arc_len_norm = (np.sqrt(2*time_factor*timesteps_norm + (deformation_norm/2)**2)
                              - (deformation_norm/2))  # (n-1,)
    # De-normalize to get actual deformation
    protruded_arc_len = protruded_arc_len_norm * protruded_len  # (n-1,)
    # Compute the wave front arc-length of the un-protruded 2d curve.
    # delta = N_open * (l_open-l_closed)
    # arc_len_open = N_open * l_closed
    # deformation = l_open / l_closed
    # Therefore:
    return protruded_arc_len/(protruded_deformation - 1)  # (n-1,)


def compute_d_director_arc_len(
        radius: np.ndarray,
        director_angle_func: Callable[..., Tuple[np.ndarray, np.ndarray]],
        angle_func_args=None) -> np.ndarray:  # ds
    angle_func_args = [] if angle_func_args is None else angle_func_args
    director_angle, _ = director_angle_func(radius, *angle_func_args)
    return 1 / np.cos(director_angle)


def compute_deformation_per_radius(
        radius: np.ndarray,
        director_arc_len: np.ndarray,  # (1, k) or (1,)
        d_director_arc_len: np.ndarray,  # (1, k) or (1,)
        timesteps: np.ndarray,
        transition_len: Number = TRANSITION_LEN,
        protruded_len: Number = PROTRUDED_LEN,
        dyn_frustum_len: Number = DYN_FRUSTUM_LEN,
        snap_slope: Number = SNAP_SLOPE,
        viscosity: Number = VISCOSITY,
        inner_rad: Number = INNER_RAD,
        inlet_pressure: Number = INLET_PRESSURE,
        critical_pressure: Number = CRITICAL_PRESSURE,
        protruded_deformation: Number = PROTRUDED_DEFORMATION
        ) -> np.ndarray:
    # Compute the protruded arc-length of the straws for all timesteps
    protruded_arc_len = compute_protruded_arc_len(
        timesteps, protruded_len, snap_slope, dyn_frustum_len,
        viscosity, inner_rad, inlet_pressure, critical_pressure,
        protruded_deformation)  # (n-1,)
    deformation, d_deformation_ds = (
            compute_deformation_per_arc_len(director_arc_len, protruded_arc_len,
                                            transition_len, protruded_deformation,))

    d_deformation = d_deformation_ds * d_director_arc_len
    return deformation, d_deformation


def compute_deformation_per_arc_len(
        director_arc_len,  # (1,) or (1, k)
        protruded_arc_len,  # (n-1, )
        transition_len: Number = TRANSITION_LEN,
        protruded_deformation: Number = PROTRUDED_DEFORMATION,
        ):
    transition_start = protruded_arc_len - transition_len  # (n-1,)
    if len(director_arc_len.shape) > 1:
        # If this is not done, and director_arc_len.shape = (1, 1)
        # then the next computation results in a shape of (1, n-1) instead of (n-1, 1)
        transition_start = transition_start.reshape(-1, 1)
    director_arc_len_norm = (director_arc_len - transition_start) / transition_len  # (n-1, ) or (n-1, k)
    deformation_norm, d_deformation_norm_ds_norm = (
        compute_deformation_norm(director_arc_len_norm)
    )
    deformation = (deformation_norm
                   * (protruded_deformation - 1) + 1)
    d_deformation_ds = (d_deformation_norm_ds_norm
                        * (protruded_deformation - 1) / transition_len)

    return deformation, d_deformation_ds


def compute_deformation_norm(director_arc_len_norm):
    """Compute normalized deformation value.

    Args:
        director_arc_len_norm:
            Normalized arc-lengths.
            0 is the start of the transition region and 1 is the
            end of the transition region.

    Returns:
        Normalized deformation values. For arc lengths between 0 and 1,
        returns values of a polynomial function that starts at 1 and ends at 0,
        where the first and second derivatives at both end points are 0.
    """
    # Define polynomial function from 0 to 1 whose value changes from 1 to 0
    # and its 1st and 2nd derivatives at both end points is 0.
    coeffs = np.array([1, 0, 0, -10, 15, -6])
    transition_deformation_poly = np.polynomial.Polynomial(coeffs)
    deformation_norm = transition_deformation_poly(director_arc_len_norm)  # (n-1, k) or (n-1, )
    d_deformation_norm_ds_norm = transition_deformation_poly.deriv(1)(director_arc_len_norm)  # (n-1, k) or (n-1, )
    # Out of bounds values
    deformation_norm[director_arc_len_norm < 0] = 1
    deformation_norm[director_arc_len_norm > 1] = 0
    d_deformation_norm_ds_norm[np.logical_or(director_arc_len_norm < 0,
                                             director_arc_len_norm > 1)] = 0

    return deformation_norm, d_deformation_norm_ds_norm # (n-1, k) or (n-1, )


# %%
def compute_deformed_curve(deformed_cross_section_curve_x,  # (num_timesteps, num_rad_samples)
                           deformed_cross_section_curve_z,  # (num_timesteps, num_rad_samples)
                           deformed_curve_theta,  # (num_timesteps, num_rad_samples)
                           init_rotation,  # scalar
                           ):
    deformed_curve_x = (deformed_cross_section_curve_x
                        * np.cos(deformed_curve_theta
                                 - deformed_curve_theta[:, 0].reshape(-1, 1)
                                 + init_rotation))
    deformed_curve_y = (deformed_cross_section_curve_x
                        * np.sin(deformed_curve_theta
                                 - deformed_curve_theta[:, 0].reshape(-1, 1)
                                 + init_rotation))
    deformed_curve_z = deformed_cross_section_curve_z
    # output shape: (num_timesteps, 3, num_rad_samples)
    return np.stack([deformed_curve_x, deformed_curve_y, deformed_curve_z], axis=1)


def compute_deformed_curve_concentric_rotations(
        deformed_cross_section_curve_x,  # (num_timesteps, num_rad_samples)
        deformed_cross_section_curve_z,  # (num_timesteps, num_rad_samples)
        undeformed_theta,  # (1, num_rad_samples)
        rotational_deformation_angle  # (num_timesteps, num_rad_samples)
        ):
    deformed_curve_x = (deformed_cross_section_curve_x
                        * np.cos(undeformed_theta
                                 + rotational_deformation_angle
                                 - rotational_deformation_angle[:, 0].reshape(-1, 1)
                                 ))
    deformed_curve_y = (deformed_cross_section_curve_x
                        * np.sin(undeformed_theta
                                 + rotational_deformation_angle
                                 - rotational_deformation_angle[:, 0].reshape(-1, 1)
                                 ))
    deformed_curve_z = deformed_cross_section_curve_z
    # output shape: (num_timesteps, 3, num_rad_samples)
    return np.stack([deformed_curve_x, deformed_curve_y, deformed_curve_z], axis=1)


# %%
def draw_deformed_shape(fig, deformed_shape, update=False):
    fig.scene.disable_render = True
    if update and DEFORMED_SHAPE_PLOT is not None:
        DEFORMED_SHAPE_PLOT.mlab_source.set(
            x=deformed_shape[0],
            y=deformed_shape[1],
            z=deformed_shape[2])
    else:
        DEFORMED_SHAPE_PLOT = mlab.mesh(deformed_shape[0],
                                        deformed_shape[1],
                                        deformed_shape[2],
                                        figure=fig)
    fig.scene.disable_render = False


if __name__ == '__main__':
    director_angle_func = sphere_deformation_angle
    angle_func_args = [GAUSSIAN_CURVATURE, PROTRUDED_DEFORMATION]
    # director_angle_func = polynomial_angle
    # angle_func_args = [np.array(0.0001)]
    # angle_func_args = [np.array(0.1)]
    data = compute_shape(director_angle_func, angle_func_args, end_time=3.5*END_TIME)
    # Data is of shape:
    # (num_timesteps, num_coords(=3), num_rad_samples, num_theta_samples)
    deformed_shape_data = data['surface']
    timesteps = data['timesteps']

    data_z = (data['deformed_cross_section_curve_z']
              - data['deformed_cross_section_curve_z'][:, 0].reshape(-1, 1))

    relevant_idx = np.concatenate([data['deformed_cross_section_curve_x'][1:, 0] > 0, [True]])
    data_x = data['deformed_cross_section_curve_x'][relevant_idx]
    data_z = data_z[relevant_idx]

    deformed_shape_data = deformed_shape_data[relevant_idx]
    deformed_shape_data[:, -1] -= deformed_shape_data[:, -1, 0][:, np.newaxis, :]

    # Symmetric deflation (TODO: Compute deflation as well)
    deformed_shape_data_symmetric = np.concatenate(
        [deformed_shape_data, deformed_shape_data[::-1]], axis=0)

    # Director and normal curve deformations
    num_directors = 12
    normal_ratio = 2
    theta_vec = np.linspace(0, 2*np.pi, num_directors+1)[:-1]
    normal_theta_vec = np.linspace(0, 2*np.pi, normal_ratio*(num_directors)+1)[:-1]

    draw_directors = True
    draw_normals = True

    if draw_directors:
        deformed_director_theta = data['deformed_director_theta'][relevant_idx]
        deformed_director_curve_data = [
            # output shape: (num_timesteps, 3, num_rad_samples)
            compute_deformed_curve(data_x, data_z, deformed_director_theta, theta)
            for theta in theta_vec]

        # rotational_deformation_angle = data['rotational_deformation_angle'][relevant_idx]
        # deformed_director_curve_data = [
        #     compute_deformed_curve_concentric_rotations(
        #         data_x, data_z,
        #         data['director_theta'] + theta,
        #         rotational_deformation_angle)
        #     for theta in theta_vec]

        # Symmetric deflation (TODO: Compute deflation as well)
        deformed_director_curve_symmetric = [
            # (2*num_timesteps, 3, num_rad_samples)
            np.concatenate(
                [deformed_director_curve,
                 deformed_director_curve[::-1]], axis=0)
            for deformed_director_curve in deformed_director_curve_data]

        # Start index of each curve
        director_curve_len_cumsum = (
            np.cumsum([
                curve_data.shape[-1]
                for curve_data in deformed_director_curve_data])
            - deformed_director_curve_data[0].shape[-1])
        # Line connection arrays
        director_connections = [
            (np.vstack(
                [np.arange(curve_start_idx,
                           curve_start_idx + curve_data.shape[-1] - 1.5),
                 np.arange(curve_start_idx + 1,
                           curve_start_idx + curve_data.shape[-1] - .5)])
             .T)
            for curve_start_idx, curve_data in
            zip(director_curve_len_cumsum, deformed_director_curve_data)]

    if draw_normals:
        # deformed_normal_curve_data = [
        #     compute_deformed_curve(
        #         data_x, data_z,
        #         data['deformed_normal_theta'],
        #         theta)
        #     for theta in normal_theta_vec]

        deformed_normal_curve_data = [
            compute_deformed_curve_concentric_rotations(
                data_x, data_z,
                data['normal_curve_theta']
                + theta
                - data['normal_curve_theta'][:, 0],
                deformed_director_theta
                - deformed_director_theta[0, :].reshape(1, -1))
            for theta in normal_theta_vec]

        # deformed_normal_curve_data = [
        #     compute_deformed_curve_concentric_rotations(
        #         data_x,
        #         data_z,
        #         data['normal_curve_theta'] + theta,
        #         rotational_deformation_angle)
        #     for theta in normal_theta_vec]

        # Symmetric deflation (TODO: Compute deflation as well)
        deformed_normal_curve_symmetric = [
            np.concatenate(
                [deformed_normal_curve,
                 deformed_normal_curve[::-1]], axis=0)
            for deformed_normal_curve in deformed_normal_curve_data]

        # Start index of each curve
        normal_curve_len_cumsum = (
            np.cumsum([
                curve_data.shape[-1]
                for curve_data in deformed_normal_curve_data])
            - deformed_normal_curve_data[0].shape[-1])
        # Line connection arrays
        normal_connections = [
            (np.vstack(
                [np.arange(curve_start_idx,
                           curve_start_idx + curve_data.shape[-1] - 1.5),
                 np.arange(curve_start_idx + 1,
                           curve_start_idx + curve_data.shape[-1] - .5)])
             .T)
            for curve_start_idx, curve_data in
            zip(normal_curve_len_cumsum, deformed_normal_curve_data)]

    # Plot animation
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)  # enable high dpi scaling
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)  # use high dpi icons
    os.environ['ETS_TOOLKIT'] = 'qt5'
    os.environ['QT_API'] = 'pyqt5'

    fig = mlab.figure("shape_deformation")
    mlab.clf()
    fig.scene.background = Color("black").rgb
    fig.scene.foreground = Color("black").rgb

    # draw_deformed_shape(fig, deformed_shape_data[0])
    start_idx = 0

    deformed_shape_plot = mlab.mesh(
        deformed_shape_data_symmetric[start_idx, 0],
        deformed_shape_data_symmetric[start_idx, 1],
        deformed_shape_data_symmetric[start_idx, 2],
        figure=fig,
        color=Color('black').rgb,
        opacity=0.6)

    if draw_directors:
        # Create the points
        director_curve_stack = np.hstack(
            # (3, num_rad_samples)
            [deformed_director_curve[start_idx]
             for deformed_director_curve in deformed_director_curve_symmetric])

        director_src = mlab.pipeline.scalar_scatter(
            director_curve_stack[0],
            director_curve_stack[1],
            director_curve_stack[2])

        # Connect them
        director_connection_stack = np.vstack(director_connections)
        director_src.mlab_source.dataset.lines = director_connection_stack
        director_src.update()

        # The stripper filter cleans up connected lines
        director_lines = mlab.pipeline.stripper(director_src)

        # Create tubes
        director_tubes = mlab.pipeline.tube(director_lines,
                                            tube_radius=1e-2)
        # Draw tubes
        director_surfs = mlab.pipeline.surface(director_tubes,
                                               color=Color("lime").rgb,
                                               figure=fig)

    if draw_normals:
        # Create the points
        normal_curve_stack = np.hstack(
            # (3, num_rad_samples)
            [deformed_normal_curve[start_idx]
             for deformed_normal_curve in deformed_normal_curve_symmetric])

        normal_src = mlab.pipeline.scalar_scatter(
            normal_curve_stack[0],
            normal_curve_stack[1],
            normal_curve_stack[2])

        # Connect them
        normal_connection_stack = np.vstack(normal_connections)
        normal_src.mlab_source.dataset.lines = normal_connection_stack
        normal_src.update()

        # The stripper filter cleans up connected lines
        normal_lines = mlab.pipeline.stripper(normal_src)

        # Create tubes
        normal_tubes = mlab.pipeline.tube(normal_lines,
                                          tube_radius=3e-3)
        # Draw tubes
        normal_surfs = mlab.pipeline.surface(normal_tubes,
                                             color=Color("white").rgb,
                                             figure=fig)

    @mlab.animate(delay=100)
    def anim():
        epochs = 100
        for epoch in range(epochs):
            for i in range(deformed_shape_data_symmetric.shape[0]):
                fig.scene.disable_render = True
                # Update the graphics
                # draw_deformed_shape(fig, deformed_shape, update=True)
                deformed_shape_plot.mlab_source.set(
                    x=deformed_shape_data_symmetric[i, 0],
                    y=deformed_shape_data_symmetric[i, 1],
                    z=deformed_shape_data_symmetric[i, 2])

                if draw_directors:
                    director_curve_stack = np.hstack(
                        [deformed_director_curve[i]
                         for deformed_director_curve in deformed_director_curve_symmetric])

                    director_surfs.mlab_source.set(
                        x=director_curve_stack[0],
                        y=director_curve_stack[1],
                        z=director_curve_stack[2])

                if draw_normals:
                    normal_curve_stack = np.hstack(
                        [normal_director_curve[i]
                         for normal_director_curve in deformed_normal_curve_symmetric])

                    normal_surfs.mlab_source.set(
                        x=normal_curve_stack[0],
                        y=normal_curve_stack[1],
                        z=normal_curve_stack[2])

                fig.scene.disable_render = False
                yield
                gc.collect(generation=1)

    anim()
    mlab.show()
