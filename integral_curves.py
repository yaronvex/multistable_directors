import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path
from typing import Tuple
from numbers import Number

# If you want to save the file to open in ipe, change the following to True.
# Note that you need to have the matplotlib ipe backend files in your directory.
# These can be downloaded from https://github.com/otfried/ipe-tools/tree/master/matplotlib.
SAVE_TO_IPE = False

if SAVE_TO_IPE:
    matplotlib.use('module://backend_ipe')

NUM_FRUSTA = 78
MIN_LENGTH = 80.  # mm
MAX_LENGTH = 415.  # mm
DEF_MIN_RAD = 55.  # mm
DEF_MAX_RAD = DEF_MIN_RAD * 1.5
DEF_NUM_CURVES = 6
FRUSTUM_DEFORM = (MAX_LENGTH - MIN_LENGTH) / NUM_FRUSTA
FRUSTUM_STATIC_LEN = (MAX_LENGTH + MIN_LENGTH) / (2*NUM_FRUSTA)
FRUSTUM_DYNAMIC_LEN = (MAX_LENGTH - MIN_LENGTH) / (2*NUM_FRUSTA)
CONNECTOR_LENGTH = 28/2
LINK_BALL_DIAM = 3.25  # mm
MIN_LINK_LENGTH = LINK_BALL_DIAM * 2

SAVE_DIR = Path(__file__).parent / 'results'
SAVE_DIR.mkdir(parents=True, exist_ok=True)
DEF_CURVE_DATA_FILENAME = 'straw_curves'
DEF_CONJ_CURVE_DATA_FILENAME = 'conjugate_curves'
DEF_INTERSECTION_DATA_FILENAME = 'fixed_points_straw'

DRAW_CONJ = True
DRAW_INTERSECTIONS = True
DRAW_LINKS = True
DRAW_GRID = False
DRAW_AXES = False

DEF_RTOL = 1e-8
DEF_ATOL = 1e-8
DEF_NUM_EVAL_POINTS = 100
# number of variables to integrate - angle of spiral,
# angle of conjugate spiral, and spiral arc-length.
Y_LEN = 3
# Indexes of the variables in the solution vector:
THETA_IDX = 0
CONJ_IDX = 1
LENGTH_IDX = 2
# Intersection marker color and transparency
INTERSECT_COLOR = (.5, .5, .5, .8)


def integral_curve(alpha_func, ri, rf, theta_i,
                   num_eval_points=DEF_NUM_EVAL_POINTS, args=None,
                   events=None):
    """Compute integral curve of spiral

    Args:
        alpha_func (object): Angle function of radius function
                             from radial direction to spiral.
        ri (Number): Initial radius of spiral.
        rf (Number): Final radius of spiral.
        theta_i (Number): Initial angle from x axis to spiral
        num_eval_points (int, optional): Number of evaluation points of the
                                         spiral curve. Defaults to
                                         DEF_NUM_EVAL_POINTS.
        args (tuple, optional): Additional arguments for alpha function.
        events (function or list, optional): Event functions

    Returns:
        tuple: Coordinates of spiral curve points as (x, y) and polar
               coordinates (r, theta) of points for which the event functions
               are equal to zero.
    """
    r_span = (ri, rf)
    r_eval = np.linspace(ri, rf, num_eval_points)
    sol = solve_ivp(calc_dy, r_span, np.array(theta_i), t_eval=r_eval,
                    args=(alpha_func, *args), rtol=DEF_RTOL, atol=DEF_ATOL,
                    vectorized=True, events=events,
                    # max_step=1e-4,
                    )
    r: np.ndarray = sol.t
    theta: np.ndarray = sol.y[[THETA_IDX, CONJ_IDX]]
    x: np.ndarray = r * np.cos(theta)
    y: np.ndarray = r * np.sin(theta)
    r_events: np.ndarray = sol.t_events
    y_events: np.ndarray = sol.y_events
    return x, y, r_events, y_events


def calc_dy(r, theta, alpha_func, *args):
    """Calculate the derivative of the angle of a spiral
    defined by the spiral angle function for a given radius

    Args:
        r (Number or ndarray): Radius of the spiral point of shape (N, )
        theta (list, ndarray): Angles of the spiral point and the
                               conjugate spiral point
        alpha_func (function): Spiral angle function
        args (list): additional arguments for alpha_func

    Returns:
        ndarray: The derivatives of the spiral angle, the conjugate
                 spiral angle, and the spiral arc-length at the given points.
    """
    if type(r) is not np.ndarray:
        r = np.array([r])

    d_y = np.zeros((Y_LEN, len(r)))
    alpha = alpha_func(r, theta, *args)
    d_y[[THETA_IDX, CONJ_IDX]] = np.tan([alpha, alpha - np.pi/2]) / r
    d_y[LENGTH_IDX] = 1 / np.cos(alpha)
    return d_y


def intersection_event(r, y, *args, th2_0):
    """Find angle difference between two curves in polar coordinates.
    Event function for finding intersections

    Args:
        r (Number): Radius at which to calculate the angle difference
        y (list, ndarray): Current solution vector containing the
                           angles of the two curves.
        th2_0 (Number): Starting angle of the second curve

    Returns:
        Number: Angle difference between curves at given point, normalized
               so it will change signs at each intersection of the curves.
    """
    theta_1 = y[THETA_IDX]
    theta_2 = y[CONJ_IDX] + th2_0
    delta_theta = theta_1 - theta_2
    delta_theta = ((-1)**(np.abs(delta_theta)//(2*np.pi))
                   * np.fmod(delta_theta, 2*np.pi))
    return delta_theta


def arc_len_event(r, y, *args, arc_len, length_idx=LENGTH_IDX):
    """Find arc-length difference between curve and given arc-length.
    Use as an event function to find points of length 's' along integral curve.

    Args:
        r (Number): Radius at which to calculate the angle difference
        y (list, ndarray): Current solution vector containing the
                           angles and length of the current integral curve.
        arc_len (Number): Desired arc-length value.

    Returns:
        Number: Arc-length difference between the integral curve and the desired arc-length
    """
    arc_len_integral = y[length_idx]
    return arc_len - arc_len_integral


def axisymmetric_integral_curves(alpha_func, ri=DEF_MIN_RAD,
                                 rf=DEF_MAX_RAD, num_eval_points=None,
                                 args=None,
                                 num_of_curves: int = DEF_NUM_CURVES,
                                 conj_ratio: int = DEF_NUM_CURVES,
                                 max_len=None):
    """Calculate and draw axisymmetric integral curves for a given spiral angle
    function. Calculate conjugate integral curves that intersect the original
    curves at a 90 degree angle. Calculate the intersection points between the
    curves with the conjugate curves and mark the intersections on the figure.

    Args:
        alpha_func (function): Spiral angle function
        ri (Number, optional): Initial radius. Defaults to DEF_MIN_RAD.
        rf (Number, optional): Final radius. Defaults to DEF_MAX_RAD.
        num_eval_points (int, optional): Number of evaluation points for each
                                         spiral curve. Defaults to None.
        args (list or tuple, optional): Additional arguments for alpha_func.
                                        Defaults to None.
        num_of_curves (int, optional): Number of spiral curves to draw.
                                       Defaults to DEF_NUM_CURVES.
        conj_ratio (int, optional): Ratio between the number of conjugate curves
                                    to the number of integral curves. Defaults to 1.

    Returns:
        np.ndarray: Data of integral curves.
        np.ndarray: Data of conjugate integral curves.
        np.ndarray: Data of intersection points between curves and conjugate curves.
        np.ndarray: Arc-length values of intersection points along the integral curve.
        np.ndarray: Vector of start angle values of integral curves.
        np.ndarray: Vector of x-axis value of intersection points on a theta_start=0 integral curve.
        np.ndarray: Vector of y-axis value of intersection points on a theta_start=0 integral curve.
        np.ndarray: Vector of r-axis value of intersection points on a theta_start=0 integral curve.
        np.ndarray: Vector of theta-axis value of intersection points on a theta_start=0 integral curve.
    """
    # Calculate integral spiral curve assuming starting angle of 0
    theta_vec, intersect_theta_vec, curve_0_x, curve_0_y, r_events, y_events =\
        zero_angle_integral_curve(
            alpha_func, ri, rf, num_eval_points, args, num_of_curves, conj_ratio, max_len)
    spiral_x, conj_x = curve_0_x
    spiral_y, conj_y = curve_0_y
    # Calculate x,y coordinates of intersections between conjugate spiral
    # curves and the 0 starting angle curve
    intersect_x, intersect_y, s_events, intersect_r, intersect_theta = events2points(r_events, y_events)
    # Calculate data of all rotated curves
    curve_data = rot_curves(spiral_x, spiral_y, theta_vec)
    conj_data = rot_curves(conj_x, conj_y, intersect_theta_vec)
    intersection_data = rot_curves(intersect_x, intersect_y, theta_vec)

    return curve_data, conj_data, intersection_data, s_events, theta_vec, \
        intersect_x, intersect_y, intersect_r, intersect_theta


def zero_angle_integral_curve(alpha_func, ri=DEF_MIN_RAD, rf=DEF_MAX_RAD,
                              num_eval_points=None, args=None,
                              num_of_curves=DEF_NUM_CURVES,
                              conj_ratio: int = DEF_NUM_CURVES,
                              max_len=None):
    """Compute an integral spiral curve starting and angle theta = 0

    Args:
        alpha_func (function): Spiral angle function
        ri (Number, optional): Initial radius. Defaults to MIN_RAD.
        rf (Number, optional): Final radius. Defaults to DEF_MAX_RAD.
        num_eval_points (int, optional): Number of evaluation points for each
                                         spiral curve. Defaults to None.
        args (list or tuple, optional): Additional arguments for alpha_func.
                                        Defaults to None.
        num_of_curves (int, optional): Number of spiral curves to draw.
                                       Defaults to DEF_NUM_CURVES.

    Returns:
        tuple: Tuple of:
            ndarray: Vector of starting angles of curves.
                     Shape: (num_of_curves, )
            ndarray: Spiral and conj. spiral x values.
                     Shape: (2, num_eval_points)
            ndarray: Spiral and conj. spiral y values.
                     Shape: (2, num_eval_points)
            list: List of intersection radii between the spiral and all
                  conjugate spirals.
            list: List of intersection angle and arc-lengths between the
                  spiral and all conjugate spirals.
    """
    # Compute vector of starting angle values
    if num_of_curves == 1:
        theta_vec = np.array([0])
    else:
        theta_vec = np.linspace(0, 2*np.pi, num_of_curves+1)[:-1]
    # Create a list of intersection event functions for each conjugate spiral
    num_of_conj = conj_ratio*num_of_curves
    intersect_theta_vec = np.linspace(0, 2*np.pi, num_of_conj+1)[:-1]
    events = [
        lambda r, y, *args, th2_0=th2_0: (
            intersection_event(r, y, *args, th2_0=th2_0)) for th2_0 in intersect_theta_vec]

    if max_len is not None:
        def max_len_event(r, y, *args, arc_len=max_len):
            return arc_len_event(r, y, *args, arc_len=arc_len)
        max_len_event.terminal = True
        events.append(max_len_event)

    # Compute integral curves and intersections
    sol = integral_curve(
        alpha_func, ri, rf, [0, 0, 0], num_eval_points, args=args,
        events=events)
    # Extract solution
    curve_0_x, curve_0_y, r_events, y_events = sol

    if max_len is not None:
        r_events = r_events[:-1]
        y_events = y_events[:-1]

    return theta_vec, intersect_theta_vec, curve_0_x, curve_0_y, r_events, y_events


def events2points(r_events, y_events):
    """Compute (x,y) coordinates of event points along integral curve

    Args:
        r_events (list of ndarrays): List of radius values of events
        y_events (list of ndarrays): List of angle values of events

    Returns:
        np.ndarray: Cartesian coordinate events of shape (2, number of events)
        np.ndarray: Vector of arc-length locations of events along the curve.
        np.ndarray: Vector of r-coordinate values of the events.
        np.ndarray: Vector of theta-coordinate values of the events.
    """
    # Find all valid intersections
    y_valid = []
    r_valid = []
    x_event = None
    y_event = None
    s_events = None
    r_event_vec = None
    theta_event_vec = None

    for y_event, r_event in zip(y_events, r_events):
        if y_event.size > 1:
            r_valid.append(r_event)
            y_valid.append(y_event)

    if len(r_valid) > 0:
        r_event_vec = np.concatenate(r_valid).reshape(1, -1)
        y_event_vec = np.concatenate(y_valid).reshape(-1, Y_LEN).T
        theta_event_vec = y_event_vec[THETA_IDX]
        s_events = np.sort(y_event_vec[LENGTH_IDX])
        x_event = r_event_vec * np.cos(theta_event_vec)
        y_event = r_event_vec * np.sin(theta_event_vec)

    return x_event, y_event, s_events, r_event_vec, theta_event_vec


def rot_curves(curve_0_x: np.ndarray, curve_0_y: np.ndarray, theta_vec: np.ndarray):
    """Calculate curve data for rotated curves.

    Args:
        curve_0_x (ndarray): x values of curve.
        curve_0_y (ndarray): y values of curve.
        theta_vec (ndarray): vector of rotation angles.

    Returns:
        ndarray: Data of all rotation curves of shape:
                 (num_of_curves, 2, num_of_eval_points)
    """
    # Concatenate x and y to one matrix representing the 0 angle curve.
    # curve_0 = [p_0, p_1, ..., p_n] where each p_i is [[x_i],[y_i]]
    # the shape of curve_0 is therefore (2, num_eval_points)
    curve_0 = np.concatenate(
        (curve_0_x.reshape(1, -1),  curve_0_y.reshape(1, -1)), axis=0)
    # Create rotation matrix slices for each initial angle in theta_vec.
    # The length of theta_vec is num_of_curves.
    # rot_mat = [R_1,R_2,...R_n] where each R_i is a rotation matrix:
    # R_i = [[cos(theta_i), -sin(theta_i)]
    #        [sin(theta_i),  cos(theta_i)]]
    # The shape of rot_mat is (num_of_curves, 2, 2)
    rot_mat = np.array([[np.cos(theta_vec), -np.sin(theta_vec)],
                        [np.sin(theta_vec), np.cos(theta_vec)]])
    rot_mat = np.moveaxis(rot_mat, -1, 0)
    # Get all rotated curves by matrix multiplication of the rotation
    # matrices and the zero angle curve.
    # The shape of the returned rotated curve data is:
    # (num_of_curves, 2, num_eval_points)
    return rot_mat @ curve_0


def place_connectors(alpha_func,
                     theta_vec: np.ndarray,
                     s_events: np.ndarray,
                     connect_len: Number = CONNECTOR_LENGTH,
                     ri: Number = DEF_MIN_RAD,
                     rf: Number = DEF_MAX_RAD,
                     num_elements: int = NUM_FRUSTA,
                     l_static: Number = FRUSTUM_STATIC_LEN,
                     l_dyn: Number = FRUSTUM_DYNAMIC_LEN,
                     min_link_len: Number = MIN_LINK_LENGTH,
                     args: list = None,
                     conj_ratio: int = 1,):
    """Find best places to place connectors given the theoretical intersection
    points of the straw integral curves and their conjugate curves.

    Args:
        alpha_func (callable): Function for calculating angle alpha of the
                               director given (r,theta).
        theta_vec (np.ndarray): Vector of starting angles of straws.
        s_events (np.ndarray): Arc-length values of intersection.
        connect_len (Number, optional): Straw connector length.
                                       Defaults to CONNECTOR_LENGTH.
        ri (Number, optional): Initial radius. Defaults to DEF_MIN_RAD.
        rf (Number, optional): Final radius. Defaults to DEF_MAX_RAD.
        num_elements (int, optional): Number of bi-stable elements in the straw.
                                      Defaults to NUM_FRUSTA.
        l_static (Number, optional): Length of the static frustum of the straw.
                                    Defaults to FRUSTUM_STATIC_LEN.
        l_dyn (Number, optional): Length of the dynamic frustum of the straw.
                                 Defaults to FRUSTUM_DYNAMIC_LEN.
        min_link_len (Number, optional): Minimum feasible link length.
                                        Defaults to MIN_LINK_LENGTH.
        args (list or tuple, optional): Additional arguments for alpha_func.
                                        Defaults to None.
        conj_ratio (int, optional): Ratio between the number of conjugate curves
                                    to the number of integral curves. Defaults to 1.

    Returns:
        np.ndarray: Link vectors (x,y) values.
                    shape = (num_of_curves, num_of_intersections, 2)
        np.ndarray: Link lengths.
                    shape = (num_of_curves, num_of_intersections)
        np.ndarray: Left-side connection points (x,y) values.
                    shape = (num_of_curves, num_of_intersections, 2)
        np.ndarray: Right-side connection points (x,y) values.
                    shape = (num_of_curves, num_of_intersections, 2)
        list: List of arc-lengths to connectors.
        list: List of number of closed frusta between two connectors.
    """
    # Initialize convergence flag and arc-length list
    converge = False
    approx_arc_len = s_events
    while not converge:
        # Compute the best locations for the connectors.
        s_new = []  # Arc-length values for connectors.
        n_new = []  # Number of retracted frusta between two connectors.
        n_acc = []  # Accumulated number of frusta for each connector
                    # (absolute location on straw).
        s_new_last = 0  # Last value of connector arc_length.
        n_acc_last = 0  # The frusta on which the last connector is installed.
        for s in approx_arc_len:
            # Theoretical number of retracted elements between connectors (Number).

            # OLD:
            #   Take into account that each connection leads to an open frusta.
            #   Compute the new minimal start length and average deformation:
            #   The approximate arc-length of the current connection
            #   minus the arc-length of the previous connection
            #   minus the length of an open frustum (static length + dynamic length)
            #   divided by the length of a retracted frustum (static length - dynamic length)
            # n_theory = (s - s_new_last - (l_static + l_dyn)) / (l_static - l_dyn)

            # UPDATED:
            #   The approximate arc-length of the current connection
            #   minus the arc-length of the previous connection
            #   divided by the length of a retracted frustum (static length - dynamic length)
            #   minus 1 (current frustum)
            n_theory = (s - s_new_last) / (l_static - l_dyn) - 1

            # Rounded number of retracted elements between connectors.
            n_minus, n_plus = np.floor(n_theory), np.ceil(n_theory)
            # Find the number of retracted elements that result in the lowest
            # arc-length error. Filter out infeasible numbers
            if n_plus < 0 or n_minus >= (num_elements - n_acc_last):
                s_new.append(np.nan)
                continue
            elif n_minus < 0:
                n_best = n_plus
            elif n_plus >= (num_elements - n_acc_last):
                n_best = n_minus
            else:
                n_list = [n_minus, n_plus]
                # OLD
                # err = [np.abs(s - (s_new_last + n*(l_static-l_dyn) + (l_static+l_dyn))) for n in n_list]
                # UPDATED
                err = [np.abs(s - (s_new_last + (n + 1) * (l_static - l_dyn))) for n in n_list]
                n_best = n_list[np.argmin(err)]
            # Compute the current connector arc_length, and the accumulated number of frusta.
            # OLD
            # s_new_last = s_new_last + n_best*(l_static-l_dyn) + (l_static+l_dyn)
            # UPDATED
            s_new_last = s_new_last + (n_best + 1) * (l_static - l_dyn)
            n_acc_last = n_acc_last + n_best + 1
            # Add the current connector data to the lists.
            s_new.append(s_new_last)
            n_new.append(n_best)
            n_acc.append(n_acc_last)

        # Find new connector locations
        x_connect, y_connect, r_connect, theta_connect = \
            integral_curve_arc_length_locations(alpha_func, s_new, ri, rf, args)
        # Compute link lengths
        links, link_len, connect_left, connect_right = compute_links(
            connect_len, alpha_func, theta_vec, x_connect, y_connect,
            r_connect, theta_connect, args, conj_ratio)
        # Filter infeasible links
        print(f'{link_len[0] = }')
        link_len_0 = link_len[0]
        if any(link_len_0 < min_link_len):
            short_link_idx = np.arange(len(link_len_0))[link_len_0 < min_link_len]
            approx_arc_len = np.delete(approx_arc_len, short_link_idx)
        else:
            # If all links are feasible, the computation has converged
            converge = True

    return links, link_len, connect_left, connect_right, s_new, n_new, n_acc


def integral_curve_arc_length_locations(alpha_func, arc_lengths, ri=DEF_MIN_RAD, rf=DEF_MAX_RAD, args=None):
    """Find locations of given arc-lengths for an integral curve.

    Args:
        alpha_func (callable): Function for calculating angle alpha of the
                               director given (r,theta).
        arc_len_list (list): list of arc-length values.
        ri (Number, optional): Initial radius. Defaults to DEF_MIN_RAD.
        rf (Number, optional): Final radius. Defaults to DEF_MAX_RAD.
        args (list or tuple, optional): Additional arguments for alpha_func.
                                        Defaults to None.

    Returns:
        np.ndarray: Vector of x coordinates of the points corresponding the
                    input arc-lengths.
        np.ndarray: Vector of y coordinates of the points corresponding the
                    input arc-lengths.
        np.ndarray: Vector of r coordinates of the points corresponding the
                    input arc-lengths.
        np.ndarray: Vector of theta coordinates of the points corresponding the
                    input arc-lengths.
    """
    if not isinstance(arc_lengths, list):
        arc_lengths = [arc_lengths]

    arc_lengths = np.array(arc_lengths)
    # Create arc-length events
    events = [lambda r, y, *args, arc_len=arc_len: (
            arc_len_event(r, y, *args, arc_len=arc_len))
              for arc_len in arc_lengths[~np.isnan(np.array(arc_lengths))]]
    # Compute integral curves and intersections
    sol = integral_curve(alpha_func, ri, rf, [0, 0, 0], num_eval_points,
                         args=args, events=events)
    # Extract event solution
    _, _, r_events, y_events = sol
    x_loc, y_loc, _, r_loc, theta_loc = events2points(r_events, y_events)

    x_loc_tot = np.ones_like(arc_lengths) * np.nan
    y_loc_tot = np.ones_like(arc_lengths) * np.nan
    r_loc_tot = np.ones_like(arc_lengths) * np.nan
    theta_loc_tot = np.ones_like(arc_lengths) * np.nan

    x_loc_tot[~np.isnan(arc_lengths)] = x_loc.squeeze()
    y_loc_tot[~np.isnan(arc_lengths)] = y_loc.squeeze()
    r_loc_tot[~np.isnan(arc_lengths)] = r_loc.squeeze()
    theta_loc_tot[~np.isnan(arc_lengths)] = theta_loc.squeeze()

    return x_loc_tot, y_loc_tot, r_loc_tot, theta_loc_tot


def compute_links(connect_len: Number,
                  alpha_func,
                  theta_vec: np.ndarray,
                  intersect_x: np.ndarray,
                  intersect_y: np.ndarray,
                  intersect_r: np.ndarray = None,
                  intersect_theta: np.ndarray = None,
                  args: list = None,
                  conj_ratio: int = 1):
    """Compute links between straws.

    Args:
        connect_len (Number): Straw connector length.
        alpha_func (callable): Function for calculating angle alpha of the
            director given (r,theta)
        theta_vec (np.ndarray): Vector of starting angles of straws
        intersect_x (np.ndarray): X value of intersection points on a
            theta_start=0 straw.
        intersect_y (np.ndarray): Y value of intersection points on a
            theta_start=0 straw.
        intersect_r (np.ndarray, optional): R value of intersection points on a
            theta_start=0 straw. If None, the function computes it using (x,y)
            data. Defaults to None.
        intersect_theta (np.ndarray, optional): Theta value of intersection
            points on a theta_start=0 straw. If None, the function computes it
            using (x,y) data. Defaults to None.
        args (list, optional): Additional arguments for alpha function.
            Defaults to None.

    Returns:
        tuple: Tuple of:
            np.ndarray: Link vectors (x,y) values.
                        shape = (num_of_curves, num_of_intersections, 2)
            np.ndarray: Link lengths.
                        shape = (num_of_curves, num_of_intersections)
            np.ndarray: Left-side connection points (x,y) values.
                        shape = (num_of_curves, num_of_intersections, 2)
            np.ndarray: Right-side connection points (x,y) values.
                        shape = (num_of_curves, num_of_intersections, 2)
    """
    # prepare data
    # (x,y) coordinates of all intersection points.
    # shape = (num_of_curves, 2, num_of_intersections)
    intersect_data = rot_curves(intersect_x, intersect_y, theta_vec)
    # Transpose to (num_of_curves, num_of_intersections, 2, 1)
    intersect_data = intersect_data.transpose((0, 2, 1))
    intersect_data = intersect_data[..., np.newaxis]
    # TODO: compute (r,theta) using (x,y) data if they are none
    # Expand intersect_theta to shape (num_of_curves, num_of_intersections)
    theta = intersect_theta.reshape(1, -1) + theta_vec.reshape(-1, 1)
    r = np.kron(np.ones((len(theta_vec), 1)), intersect_r.reshape(1, -1))
    # Compute angle alpha for all values of r and theta
    alpha = alpha_func(r, theta) if args is None else alpha_func(r, theta, *args)
    # Rotation matrix from radial direction to normal direction
    rot_angle = alpha + np.pi/2
    rot_norm = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
                         [np.sin(rot_angle), np.cos(rot_angle)]])
    rot_norm = np.moveaxis(rot_norm, [0, 1], [-2, -1])
    # compute left and right connection points.
    # We use np.squeeze to drop the last dimension and get:
    # shape = (num_of_curves, num_of_intersections, 2)
    r_direction = intersect_data/r[..., np.newaxis, np.newaxis]
    connect_vec = connect_len * rot_norm @ r_direction
    connect_left = np.squeeze(intersect_data + connect_vec)
    connect_right = np.squeeze(intersect_data - connect_vec)
    # Compute link vectors.
    # Links connect the i_th right connection in the j_th curve
    # to the (i+1)th left connection on the (j+1)th curve.
    # We use np.roll to wrap the last curve back to curve 0.
    # shape = (num_of_curves, num_of_intersections - conj_ratio, 2)
    links = np.roll(connect_left[:, conj_ratio:], shift=1, axis=0) - connect_right[:, :-conj_ratio]
    # Compute link lengths
    link_len = np.sqrt(links[..., 0]**2 + links[..., 1]**2)
    # return link and connections data
    return links, link_len, connect_left, connect_right


def draw_links(connect_left: np.ndarray,
               connect_right: np.ndarray,
               links: np.ndarray,
               fig=None, ax=None,
               conj_ratio: int = 1):
    """Draw links and connectors

    Args:
        connect_left (np.ndarray): Vector of left side connector points
        connect_right (np.ndarray): Vector of right side connector points
        links (np.ndarray): link data
        fig (_type_, optional): Figure to draw on. Defaults to None.
        ax (_type_, optional): Axes to draw on. Defaults to None.
        conj_ratio (int, optional): Number of conjugate curves per integral curves. Defaults to 1.

    Returns:
        Tuple[Figure, Axes]: Resultant Figure and Axes objects.
    """
    if ax is None:
        plt.ion()
        fig, ax = plt.subplots()
        ax.axis('equal')
        plt.grid(True)
    # Draw connectors as markers
    ax.plot(connect_left[:, conj_ratio:, 0].reshape(-1, 1),
            connect_left[:, conj_ratio:, 1].reshape(-1, 1),
            linestyle='',
            marker='*',
            color='k')
    ax.plot(connect_right[:, :-conj_ratio, 0].reshape(-1, 1),
            connect_right[:, :-conj_ratio, 1].reshape(-1, 1),
            linestyle='',
            marker='*',
            color='k')
    # Draw connector lines
    for left, right in zip(connect_left.reshape(-1, 2), connect_right.reshape(-1, 2)):
        ax.plot([left[0], right[0]], [left[1], right[1]], color='k')
    # Draw links
    link_x1 = connect_right[:, :-conj_ratio, 0]
    link_y1 = connect_right[:, :-conj_ratio, 1]
    link_x2 = link_x1 + links[..., 0]
    link_y2 = link_y1 + links[..., 1]
    link_x = np.concatenate([link_x1.reshape(1, -1),
                             link_x2.reshape(1, -1)], axis=0)
    link_y = np.concatenate([link_y1.reshape(1, -1),
                             link_y2.reshape(1, -1)], axis=0)
    ax.plot(link_x, link_y, color='c', linewidth=3, alpha=0.5)
    # Return figure and axes
    return fig, ax


def draw_curve_rotations(curve_data, fig=None, ax=None,
                         **kwargs):
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
    # if color is None:
    #     color = 'blue'
    # if linestyle is None:
    #     linestyle = '-'
    for curve in curve_data:
        ax.plot(curve[0], curve[1], **kwargs)
    return fig, ax


def create_dataframe(curve_data: np.ndarray, name: str = None) -> pd.DataFrame:
    """Create DataFrame from curves

    Args:
        curve_data (np.ndarray): Curve xy coordinate data.
                                 shape = (num_of_curves, 2, num_of_eval_points)
        name (str, optional): String for column descriptions. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe for the input curves
    """
    if name is None:
        name = 'curve'
    # Extract number of evaluation points
    num_eval_points = curve_data.shape[-1]
    # Reshape to (2*num_of_curves, num_of_eval_points).
    # The x and y data for each curve are in consecutive rows.
    data_2d = curve_data.reshape((-1, num_eval_points)).T
    # Name columns
    columns = [f'{name}_{i//2+1}_x' if i % 2 == 0 else f'{name}_{i//2+1}_y'
               for i in range(data_2d.shape[1])]
    # Return DataFrame
    return pd.DataFrame(data_2d, columns=columns)


def save_integral_curve_data(curve_data: np.ndarray,
                             conj_data: np.ndarray = None,
                             intersection_data: np.ndarray = None,
                             s_events: np.ndarray = None,
                             save_dir: Path = None,
                             curve_file_name: str = None,
                             conj_curve_file_name: str = None,
                             intersection_file_name: str = None,) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create dataframes of integral curve data and save as csv files.

    Args:
        curve_data (np.ndarray): Data of integral curves.
        conj_data (np.ndarray, optional): Data of conjugate integral curves.
            Defaults to None.
        intersection_data (np.ndarray, optional): Data of intersection points
            between curves and conjugate curves. Defaults to None.
        s_events (np.ndarray, optional): Arc-length values of intersection
            points along the integral curve. Defaults to None.
        save_dir (Path, optional): Path to directory for saving csv. Defaults to None.
        curve_file_name (str, optional): Integral curve csv file name. Defaults to None.
        conj_curve_file_name (str, optional): Conjugate curve csv file name. Defaults to None.
        intersection_file_name (str, optional): Intersection points csv file name. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - Dataframe of integral curve data.
            - Dataframe of conjugate curve data.
            - Dataframe of intersection points data.
    """
    save_dir = Path(save_dir) if save_dir is not None else SAVE_DIR
    curve_file_name = curve_file_name if curve_file_name is not None else DEF_CURVE_DATA_FILENAME
    curve_df = create_dataframe(curve_data, 'straw')
    curve_df.to_csv(str(save_dir / (curve_file_name+'.csv')))
    # Create dataframe and save conjugate curve data if given
    if conj_data is not None:
        conj_curve_file_name = conj_curve_file_name if conj_curve_file_name is not None \
            else DEF_CONJ_CURVE_DATA_FILENAME
        conj_df = create_dataframe(conj_data, name='conj_curve')
        conj_df.to_csv(str(save_dir / (conj_curve_file_name+'.csv')))
    # Create dataframe and save intersection points data if given
    if intersection_data is not None:
        intersection_file_name = intersection_file_name if intersection_file_name is not None \
            else DEF_INTERSECTION_DATA_FILENAME
        intersection_df = create_dataframe(intersection_data, name='fixed_points_straw')
        if s_events is not None:
            intersection_df['arc_length'] = s_events
            col_at_start = ['arc_length']
            intersection_df = intersection_df[
                (list(col_at_start) + [c for c in intersection_df if c not in col_at_start])]

        intersection_df.to_csv(str(save_dir / (intersection_file_name+'.csv')))

    return curve_df, conj_df, intersection_df


def draw_integral_curves(curve_data: np.ndarray,
                         conj_data: np.ndarray = None,
                         intersection_data: np.ndarray = None,
                         draw_conj: bool = True,
                         draw_intersections: bool = True,
                         fig=None,
                         ax=None):
    """Draw integral curves, conjugate integral curves, and intersection points between them

    Args:
        curve_data (np.ndarray): Data of integral curves.
        conj_data (np.ndarray, optional): Data of conjugate integral curves.
            Defaults to None.
        intersection_data (np.ndarray, optional): Data of intersection points
            between curves and conjugate curves. Defaults to None.
        draw_conj (bool, optional): Flag to draw conjugate curves. Defaults to True.
        draw_intersections (bool, optional): Flag to draw intersection points. Defaults to True.
        fig (Figure, optional): Figure to draw on. Defaults to None.
        ax (Axes, optional): Axes to draw on. Defaults to None.

    Returns:
        Tuple[Figure, Axes]: Resultant Figure and Axes objects.
    """
    # Draw spirals, conjugate spirals, and intersections
    fig, ax = draw_curve_rotations(curve_data, fig=fig, ax=ax, color='blue')
    if conj_data is not None and draw_conj:
        fig, ax = draw_curve_rotations(conj_data, fig=fig, ax=ax, color='red', linestyle='--')

    if intersection_data is not None and draw_intersections:
        fig, ax = draw_curve_rotations(intersection_data, fig=fig, ax=ax,
                                       color=INTERSECT_COLOR, marker='.',
                                       markersize=10,
                                       linestyle='')

    return fig, ax


def alpha_const(r, theta, alpha):
    """Compute constant spiral angle

    Args:
        r (Number or ndarray): Radius of spiral
        theta (Number or ndarray): Angle of spiral point
        alpha (Number): Spiral angle values

    Returns:
        ndarray: Spiral angle values
    """
    return alpha * np.ones_like(r)


def alpha_const_curvature(r, theta, gauss_curv, deform):
    """Compute the constant Gaussian curvature spiral angle

    Args:
        r (Number or ndarray): radius of spiral
        theta (Number or ndarray): Angle of spiral point
        gauss_curv (Number): Gaussian curvature of deformed surface
        deform (Number): Axial deformation of directors

    Returns:
        ndarray: Spiral angle values
    """
    c_k, c, _ = const_curvature_director_params(gauss_curv, deform)
    return 0.5 * np.arccos(-0.5 * c_k * r**2 + c)


def const_curvature_director_params(gauss_curv, deform):
    """Calculate parameters for constant curvature spiral angle function

    Args:
        gauss_curv (Number): Gaussian curvature of deformed surface
        deform (Number): Axial deformation of directors

    Returns:
        tuple: parameters for constant curvature spiral angle function,
               and the maximal allowed radius of the spiral.
    """
    c_k = -gauss_curv / (1 - deform**(-2))
    c = (1 - 2 / (1 + deform))
    rf_max = np.sqrt(2 * (1 - c) / -c_k)
    return c_k, c, rf_max


def polynomial_angle(radius, theta, coeffs):
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
    return angle_poly(radius)


if __name__ == '__main__':
    num_eval_points = 1000
    ri = DEF_MIN_RAD

    # Old -
    #   Take into account that each connection leads to an open frusta.
    #   Compute the new minimal start length and average deformation.
    # num_connections = 13  # Change this iteratively
    # max_retracted_len = MIN_LENGTH + num_connections * FRUSTUM_DEFORM

    # UPDATED -
    #   Start with all frusta in retracted position and use the actual deformation
    max_retracted_len = MIN_LENGTH + 0.5  # mm
    deform = MAX_LENGTH / max_retracted_len

    print(f'{max_retracted_len = } mm')
    print(f'Deformation = {deform}')
    connect_len = CONNECTOR_LENGTH

    alpha_func = alpha_const_curvature
    sphere_rad = 170  # mm
    gauss_curv = 1/(sphere_rad**2)
    args = (gauss_curv, deform)
    _, _, rf_max = const_curvature_director_params(gauss_curv, deform)
    rf = (1 - 1e-6)*rf_max
    print(f'{rf_max - ri = }')

    # alpha_func = alpha_const
    # alpha = np.pi/2-0.1
    # args = [alpha]
    # rf = 1

    alpha_func = polynomial_angle
    # coeffs = [0, 2.5*1e-3]
    coeffs = [0, 2.183*1e-3]
    args = [coeffs]
    rf = 200

    # num_of_curves = 3
    num_of_curves = 12
    conj_ratio: int = 2

    # Compute integral curve data
    curve_data, conj_data, intersection_data, s_events, theta_vec, \
        intersect_x, intersect_y, intersect_r, intersect_theta = axisymmetric_integral_curves(
            alpha_func,
            ri=ri,
            rf=rf,
            num_eval_points=num_eval_points,
            args=args,
            num_of_curves=num_of_curves,
            conj_ratio=conj_ratio,
            max_len=max_retracted_len,
            )

    # Compute links and connection points
    # links, link_len, connect_left, connect_right = compute_links(
    #     connect_len, alpha_func, theta_vec, intersect_x, intersect_y,
    #     intersect_r, intersect_theta, args, conj_ratio)

    links, link_len, connect_left, connect_right, s_new, n_new, n_acc = \
        place_connectors(alpha_func, theta_vec, s_events, connect_len, ri, rf,
                         args=args, conj_ratio=conj_ratio)

    # Save data
    save_integral_curve_data(curve_data, conj_data, intersection_data, s_events)

    # Draw integral curves
    fig, ax = draw_integral_curves(curve_data, conj_data, intersection_data,
                                   DRAW_CONJ, DRAW_INTERSECTIONS)
    # Draw links and connection points
    if DRAW_LINKS:
        fig, ax = draw_links(connect_left, connect_right, links, fig, ax, conj_ratio)

    if not DRAW_GRID:
        ax.grid(False)

    if not DRAW_AXES:
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])

    if SAVE_TO_IPE:
        # Save figure to ipe file
        fig.savefig(str(SAVE_DIR / 'flower_connections_clean.ipe'), format="ipe")
        # fig.savefig(str(SAVE_DIR / 'sphere_connections_clean.ipe'), format="ipe")

        np.set_printoptions(precision=1)
        print(f'{FRUSTUM_STATIC_LEN = }')
        print(f'{FRUSTUM_DYNAMIC_LEN = }')
        print(f'{s_events = }')
        s_new = np.array(s_new)
        print(f'{s_new[~np.isnan(s_new)] = }')
        print(f'{n_new = }')
        print(f'{n_acc = }')
        if np.any(link_len):
            print(f'link lengths: {link_len[0][~np.isnan(link_len[0])]}')

    plt.show(block=True)
