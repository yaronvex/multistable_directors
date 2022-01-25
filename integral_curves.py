import time
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path


NUM_FRUSTA = 78
MIN_LENGTH = 80.  # mm
MAX_LENGTH = 415.  # mm
MIN_RAD = 55.  # mm
FRUSTUM_DEFORM = (MAX_LENGTH - MIN_LENGTH) / NUM_FRUSTA

SAVE_DIR = Path(__file__).parent / 'results'
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def integral_curve(alpha_func, ri, rf, theta_i,
                   num_eval_points=50, args=None, events=None):
    """Compute integral curve of spiral

    Args:
        alpha_func (object): Angle function of radius function
                             from radial direction to spiral.
        ri (float): Initial radius of spiral.
        rf (float): Final radius of spiral.
        theta_i (float): Initial angle from x axis to spiral
        num_eval_points (int, optional): Number of evaluation points of the
                                         spiral curve. Defaults to 50.
        args (tuple, optional): Additional arguments for alpha function.
        events (function or list, optional): Event functions

    Returns:
        tuple: Coordinates of spiral curve points as (x, y) and polar
               coordinates (r, theta) of points for which the event functions
               are equal to zero.
    """
    r_span = (ri, rf)
    r_eval = np.linspace(ri, rf, num_eval_points)
    sol = solve_ivp(d_theta, r_span, np.array(theta_i), t_eval=r_eval,
                    args=(alpha_func, *args), rtol=1e-8, atol=1e-8,
                    vectorized=True, events=events,
                    # max_step=1e-4,
                    )
    r: np.ndarray = sol.t
    theta: np.ndarray = sol.y[:2]
    x: np.ndarray = r * np.cos(theta)
    y: np.ndarray = r * np.sin(theta)
    r_events: np.ndarray = sol.t_events
    y_events: np.ndarray = sol.y_events
    return x, y, r_events, y_events


def d_theta(r, theta, alpha_func, *args):
    """Calculate the derivative of the angle of a spiral
    defined by the spiral angle function for a given radius

    Args:
        r (float or ndarray): Radius of the spiral point
        theta (list, ndarray): Angles of the spiral point and the
                               conjugate spiral point
        alpha_func (function): Spiral angle function
        args (list): additional arguments for alpha_func

    Returns:
        ndarray: The derivative of the spiral and the conjugate
                 spiral at the given points.
    """
    d_y = np.zeros((3, len([r]))) if len([r]) > 1 else np.zeros((3, ))
    alpha = alpha_func(r, theta, *args)
    d_y[:2] = np.tan([alpha, alpha - np.pi/2]) / r
    d_y[2] = 1 / np.cos(alpha)
    return d_y


def intersection(r, theta, *args, th2_0):
    """Find angle difference between two curves in polar coordinates.
    Event function for finding intersections

    Args:
        r (float): Radius at which to calculate the angle difference
        theta (list, ndarray): Angles of the two curves
        th2_0 (float): Starting angle of the second curve

    Returns:
        float: Angle difference between curves at given point, normalized
               so it will change signs at each intersection of the curves.
    """
    delta_theta = theta[0] - (theta[1] + th2_0)
    delta_theta = (-1)**(np.abs(delta_theta)//(2*np.pi)) *\
        np.fmod(delta_theta, 2*np.pi)
    return delta_theta


def compute_axisymmetric_integral_curves(alpha_func, ri=0, rf=1,
                                         num_eval_points=None, args=None,
                                         num_of_curves=4, ax=None,
                                         draw_conj=False):
    """Calculate and draw axisymmetric integral curves for a given spiral angle
    function. Calculate conjugate integral curves that intersect the original
    curves at a 90 degree angle. Calculate the intersection points between the
    curves with the conjugate curves and mark the intersections on the figure.

    Args:
        alpha_func (function): Spiral angle function
        ri (float, optional): Initial radius. Defaults to 0.
        rf (float, optional): Final radius. Defaults to 1.
        num_eval_points (int, optional): Number of evaluation points for each
                                         spiral curve. Defaults to None.
        args (list or tuple, optional): Additional arguments for alpha_func.
                                        Defaults to None.
        num_of_curves (int, optional): Number of spiral curves to draw.
                                       Defaults to 4.
        ax (axis object, optional): matplotlib axis object. Defaults to None.
        draw_conj (bool, optional): Flag to draw the conjugate curves.
                                    Defaults to False.

    Returns:
        tuple: Figure and axis objects
    """
    theta_vec, curve_0_x, curve_0_y, r_events, y_events =\
        compute_zero_angle_integral_curve(
            alpha_func, ri, rf, num_eval_points, args, num_of_curves)

    intersections, s_events = process_intersections(r_events, y_events)
    curve_data = calc_curve_data(curve_0_x[0], curve_0_y[0], theta_vec)
    conj_data = calc_curve_data(curve_0_x[1], curve_0_y[1], theta_vec)
    intersection_data = calc_curve_data(
        intersections[0], intersections[1], theta_vec)\

    curve_df = create_dataframe(curve_data, 'straw')
    conj_df = create_dataframe(conj_data, name='conj_curve')
    intersection_df = create_dataframe(intersection_data,
                                       name='fixed_points_straw')
    curve_df.to_csv(str(SAVE_DIR / 'straw_curves.csv'))
    conj_df.to_csv(str(SAVE_DIR / 'conjugate_curves.csv'))
    intersection_df.to_csv(str(SAVE_DIR / 'fixed_points_straw.csv'))

    fig, ax = draw_curve_rotations(curve_data)
    if draw_conj:
        fig, ax = draw_curve_rotations(conj_data, fig=fig, ax=ax, color='red')

    fig, ax = draw_curve_rotations(intersection_data, fig=fig, ax=ax,
                                   color=(.5, .5, .5, .8), marker='o',
                                   linestyle='')
    fig.savefig(str(SAVE_DIR / 'director_figure.png'))
    return fig, ax, s_events


def compute_zero_angle_integral_curve(alpha_func, ri=0, rf=1,
                                      num_eval_points=None, args=None,
                                      num_of_curves=4):
    if num_of_curves == 1:
        theta_vec = np.array([0])
    else:
        theta_vec = np.linspace(0, 2*np.pi, num_of_curves+1)[:-1]
    events = [
        lambda t, y, *args, th2_0=th2_0: (
            intersection(t, y, *args, th2_0=th2_0)) for th2_0 in theta_vec]
    sol = integral_curve(
        alpha_func, ri, rf, [0, 0, 0], num_eval_points, args=args,
        events=events)
    curve_0_x, curve_0_y, r_events, y_events = sol
    return theta_vec, curve_0_x, curve_0_y, r_events, y_events


def process_intersections(r_events, y_events):
    """Compute (x,y) coordinates of intersection points

    Args:
        r_events (list of ndarrays): List of radius values of intersections
                                     between a spiral curve and all conjugate
                                     spiral curves
        y_events (list of ndarrays): List of angle values of intersections

    Returns:
        ndarray: Cartesian coordinate intersections on shape
                 (2, number of intersections)
    """
    y_valid = []
    r_valid = []
    for y_event, r_event in zip(y_events, r_events):
        if y_event.size > 1:
            r_valid.append(r_event)
            y_valid.append(y_event)

    # print(f'{y_events[valid] = }')
    r_event_vec = np.concatenate(r_valid).reshape(1, -1)
    y_event_vec = np.concatenate(y_valid).reshape(-1, 3).T
    theta_event_vec = y_event_vec[0]
    s_events = np.sort(y_event_vec[2])
    x_event = r_event_vec * np.cos(theta_event_vec)
    y_event = r_event_vec * np.sin(theta_event_vec)
    return np.concatenate([x_event, y_event], axis=0), s_events


def calc_curve_data(curve_0_x, curve_0_y, theta_vec):
    """Calculate curve data for rotated curves.

    Args:
        curve_0_x (ndarray): x values of curve.
        curve_0_y (ndarray): y values of curve.
        theta_vec (ndarray): vector of rotation angles.

    Returns:
        ndarray: Data of all rotation curves of shape:
                 (num_of_curves, 2, num_of_eval_points)
    """
    curve_0 = np.concatenate(
        (curve_0_x.reshape(1, -1),  curve_0_y.reshape(1, -1)), axis=0)
    rot_mat = np.array([[np.cos(theta_vec), -np.sin(theta_vec)],
                        [np.sin(theta_vec), np.cos(theta_vec)]])
    rot_mat = np.moveaxis(rot_mat, -1, 0)
    curve_data = rot_mat @ curve_0
    return curve_data


def draw_curve_rotations(curve_data, fig=None, ax=None,
                         color=None, marker=None, linestyle=None):
    """Draw all rotations of a given curve

    Args:
        curve_data (ndarray): Data of all rotation curves.
        fig (figure, optional): Figure object. Defaults to None.
        ax (Axis, optional): Axis object. Defaults to None.
        color (String or Tuple, optional): Color option for drawing.
                                           Defaults to None.
        marker (Marker or String, optional): Marker for drawing.
                                             Defaults to None.
        linestyle (String, optional): Line style for drawing. Defaults to None.

    Returns:
        Tuple: Current figure and axis objects
    """
    if ax is None:
        plt.ion()
        fig, ax = plt.subplots()
        ax.axis('equal')
        plt.grid(True)
    if color is None:
        color = 'blue'
    if linestyle is None:
        linestyle = '-'
    for curve in curve_data:
        ax.plot(curve[0], curve[1], color=color, marker=marker,
                linestyle=linestyle)
    return fig, ax


def create_dataframe(curve_data, name=None):
    if name is None:
        name = 'curve'
    num_eval_points = curve_data.shape[-1]
    data_2d = curve_data.reshape((-1, num_eval_points)).T
    columns = [f'{name}_{i//2+1}_x' if i % 2 == 0 else f'{name}_{i//2+1}_y'
               for i in range(data_2d.shape[1])]
    curve_df = pd.DataFrame(data_2d, columns=columns)
    return curve_df


def alpha_const(r, theta, alpha):
    """Compute constant spiral angle

    Args:
        r (float or ndarray): Radius of spiral
        theta (float or ndarray): Angle of spiral point
        alpha (float): Spiral angle values

    Returns:
        ndarray: Spiral angle values
    """
    return alpha * np.ones_like(r)


def alpha_const_curvature(r, theta, K, deform):
    """Compute the constant Gaussian curvature spiral angle

    Args:
        r (float or ndarray): radius of spiral
        theta (float or ndarray): Angle of spiral point
        K (float): Gaussian curvature of deformed surface
        deform (float): Axial deformation of directors

    Returns:
        ndarray: Spiral angle values
    """
    c_k, c, _ = const_curvature_director_params(K, deform)
    return 0.5 * np.arccos(-0.5 * c_k * r**2 + c)


def const_curvature_director_params(K, deform):
    """Calculate parameters for constant curvature spiral angle function

    Args:
        K (float): Gaussian curvature of deformed surface
        deform (float): Axial deformation of directors

    Returns:
        tuple: parameters for constant curvature spiral angle function,
               and the maximal allowed radius of the spiral.
    """
    c_k = -K / (1 - deform**(-2))
    c = (1 - 2 / (1 + deform))
    rf_max = np.sqrt(2 * (1 - c) / -c_k)
    return c_k, c, rf_max


if __name__ == '__main__':
    num_eval_points = 1000
    ri = MIN_RAD
    alpha_func = alpha_const_curvature
    sphere_rad = 183  # mm
    K = 1/(sphere_rad**2)
    # deform = 2
    num_connections = 8  # Change this iteratively
    min_length = MIN_LENGTH + num_connections * FRUSTUM_DEFORM
    print(f'{min_length = } mm')
    deform = MAX_LENGTH / min_length
    print(f'Deformation = {deform}')
    args = (K, deform)
    _, _, rf_max = const_curvature_director_params(K, deform)
    rf = (1 - 1e-6)*rf_max
    num_of_curves = 12
    # alpha_func = alpha_const
    # alpha = np.pi/2-0.1
    # args = [alpha]
    # rf = 1
    # num_of_curves = 3
    fig, ax, s_events = compute_axisymmetric_integral_curves(
        alpha_func, ri=ri, rf=rf, num_eval_points=num_eval_points, args=args,
        num_of_curves=num_of_curves, draw_conj=True)
    print(f'{s_events = }')
    plt.show(block=True)
