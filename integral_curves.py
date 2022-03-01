import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path


NUM_FRUSTA = 78
MIN_LENGTH = 80.  # mm
MAX_LENGTH = 415.  # mm
MIN_RAD = 55.  # mm
DEF_MAX_RAD = MIN_RAD * 1.5
DEF_NUM_CURVES = 6
FRUSTUM_DEFORM = (MAX_LENGTH - MIN_LENGTH) / NUM_FRUSTA
CONNECTOR_LENGTH = 28/2

SAVE_DIR = Path(__file__).parent / 'results'
SAVE_DIR.mkdir(parents=True, exist_ok=True)

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
        ri (float): Initial radius of spiral.
        rf (float): Final radius of spiral.
        theta_i (float): Initial angle from x axis to spiral
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
        r (float or ndarray): Radius of the spiral point of shape (N, )
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


def intersection(r, y, *args, th2_0):
    """Find angle difference between two curves in polar coordinates.
    Event function for finding intersections

    Args:
        r (float): Radius at which to calculate the angle difference
        y (list, ndarray): Current solution vector containing the
                           angles of the two curves.
        th2_0 (float): Starting angle of the second curve

    Returns:
        float: Angle difference between curves at given point, normalized
               so it will change signs at each intersection of the curves.
    """
    theta_1 = y[THETA_IDX]
    theta_2 = y[CONJ_IDX] + th2_0
    delta_theta = theta_1 - theta_2
    delta_theta = (-1)**(np.abs(delta_theta)//(2*np.pi)) *\
        np.fmod(delta_theta, 2*np.pi)
    return delta_theta


def compute_axisymmetric_integral_curves(alpha_func, ri=MIN_RAD,
                                         rf=DEF_MAX_RAD, num_eval_points=None,
                                         args=None,
                                         num_of_curves: int = DEF_NUM_CURVES,
                                         conj_ratio: int = DEF_NUM_CURVES,
                                         connect_len=0,
                                         ax=None,
                                         draw_conj=False):
    """Calculate and draw axisymmetric integral curves for a given spiral angle
    function. Calculate conjugate integral curves that intersect the original
    curves at a 90 degree angle. Calculate the intersection points between the
    curves with the conjugate curves and mark the intersections on the figure.

    Args:
        alpha_func (function): Spiral angle function
        ri (float, optional): Initial radius. Defaults to MIN_RAD.
        rf (float, optional): Final radius. Defaults to DEF_MAX_RAD.
        num_eval_points (int, optional): Number of evaluation points for each
                                         spiral curve. Defaults to None.
        args (list or tuple, optional): Additional arguments for alpha_func.
                                        Defaults to None.
        num_of_curves (int, optional): Number of spiral curves to draw.
                                       Defaults to DEF_NUM_CURVES.
        ax (axis object, optional): matplotlib axis object. Defaults to None.
        draw_conj (bool, optional): Flag to draw the conjugate curves.
                                    Defaults to False.

    Returns:
        tuple: Figure and axis objects
    """
    # Calculate integral spiral curve assuming starting angle of 0
    theta_vec, intersect_theta_vec, curve_0_x, curve_0_y, r_events, y_events =\
        compute_zero_angle_integral_curve(
            alpha_func, ri, rf, num_eval_points, args, num_of_curves,
            conj_ratio)
    spiral_x, conj_x = curve_0_x
    spiral_y, conj_y = curve_0_y
    # Calculate x,y coordinates of intersections between conjugate spiral
    # curves and the 0 starting angle curve
    intersections, s_events, intersect_r, intersect_theta = process_intersections(r_events, y_events)
    intersect_x, intersect_y = intersections
    # Calculate data of all rotated curves
    curve_data = calc_curve_data(spiral_x, spiral_y, theta_vec)
    conj_data = calc_curve_data(conj_x, conj_y, intersect_theta_vec)
    intersection_data = calc_curve_data(intersect_x, intersect_y, theta_vec)

    # Create dataframes from spirals, conjugate spirals, and intersections
    curve_df = create_dataframe(curve_data, 'straw')
    conj_df = create_dataframe(conj_data, name='conj_curve')
    intersection_df = create_dataframe(intersection_data,
                                       name='fixed_points_straw')
    intersection_df['arc_length'] = s_events
    col_at_start = ['arc_length']
    intersection_df = intersection_df[[c for c in col_at_start] +
                                      [c for c in intersection_df
                                       if c not in col_at_start]]
    # Save dataframes to csv files
    curve_df.to_csv(str(SAVE_DIR / 'straw_curves.csv'))
    conj_df.to_csv(str(SAVE_DIR / 'conjugate_curves.csv'))
    intersection_df.to_csv(str(SAVE_DIR / 'fixed_points_straw.csv'))
    # Draw spirals, conjugate spirals, and intersections
    fig, ax = draw_curve_rotations(curve_data)
    if draw_conj:
        fig, ax = draw_curve_rotations(conj_data, fig=fig, ax=ax, color='red', linestyle='--')

    fig, ax = draw_curve_rotations(intersection_data, fig=fig, ax=ax,
                                   color=INTERSECT_COLOR, marker='o',
                                   linestyle='')
    # Links
    link_len = []
    if connect_len:
        # Compute links and connection points
        links, link_len, connect_left, connect_right = compute_links(
            connect_len, alpha_func, theta_vec, intersect_x, intersect_y,
            intersect_r, intersect_theta, args, conj_ratio)
        # Plot links and connection points
        fig, ax = draw_links(connect_left, connect_right, links, fig, ax, conj_ratio)
        
    # Save figure to file
    fig.savefig(str(SAVE_DIR / 'director_figure.png'))
    return fig, ax, s_events, link_len


def compute_zero_angle_integral_curve(alpha_func, ri=MIN_RAD, rf=DEF_MAX_RAD,
                                      num_eval_points=None, args=None,
                                      num_of_curves=DEF_NUM_CURVES,
                                      conj_ratio: int=DEF_NUM_CURVES):
    """Compute an integral spiral curve starting and angle theta = 0

    Args:
        alpha_func (function): Spiral angle function
        ri (float, optional): Initial radius. Defaults to MIN_RAD.
        rf (float, optional): Final radius. Defaults to DEF_MAX_RAD.
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
            intersection(r, y, *args, th2_0=th2_0)) for th2_0 in intersect_theta_vec]
    # Compute integral curves and intersections
    sol = integral_curve(
        alpha_func, ri, rf, [0, 0, 0], num_eval_points, args=args,
        events=events)
    # Extract solution
    curve_0_x, curve_0_y, r_events, y_events = sol
    return theta_vec, intersect_theta_vec, curve_0_x, curve_0_y, r_events, y_events


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
    # Find all valid intersections
    y_valid = []
    r_valid = []
    for y_event, r_event in zip(y_events, r_events):
        if y_event.size > 1:
            r_valid.append(r_event)
            y_valid.append(y_event)

    r_event_vec = np.concatenate(r_valid).reshape(1, -1)
    y_event_vec = np.concatenate(y_valid).reshape(-1, Y_LEN).T
    theta_event_vec = y_event_vec[THETA_IDX]
    s_events = np.sort(y_event_vec[LENGTH_IDX])
    x_event = r_event_vec * np.cos(theta_event_vec)
    y_event = r_event_vec * np.sin(theta_event_vec)
    intersections = np.concatenate([x_event, y_event], axis=0)
    return intersections, s_events, r_event_vec, theta_event_vec


def calc_curve_data(curve_0_x: np.ndarray, curve_0_y: np.ndarray, theta_vec: np.ndarray):
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
    curve_data: np.ndarray = rot_mat @ curve_0
    return curve_data


def compute_links(connect_len: float,
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
        connect_len (float): Straw connector length.
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
    intersect_data = calc_curve_data(intersect_x, intersect_y, theta_vec)
    # Transpose to (num_of_curves, num_of_intersections, 2, 1)
    intersect_data = intersect_data.transpose((0, 2, 1))
    intersect_data = intersect_data[..., np.newaxis]
    # Expand intersect_theta to shape (num_of_curves, num_of_intersections)
    theta = intersect_theta.reshape(1, -1) + theta_vec.reshape(-1, 1)
    r = np.kron(np.ones((len(theta_vec), 1)), intersect_r.reshape(1, -1))
    # Compute angle alpha for all values of r and theta
    if args is not None:
        alpha = alpha_func(r, theta, *args)
    else:
        alpha = alpha_func(r, theta)
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


def alpha_const_curvature(r, theta, gauss_curv, deform):
    """Compute the constant Gaussian curvature spiral angle

    Args:
        r (float or ndarray): radius of spiral
        theta (float or ndarray): Angle of spiral point
        gauss_curv (float): Gaussian curvature of deformed surface
        deform (float): Axial deformation of directors

    Returns:
        ndarray: Spiral angle values
    """
    c_k, c, _ = const_curvature_director_params(gauss_curv, deform)
    return 0.5 * np.arccos(-0.5 * c_k * r**2 + c)


def const_curvature_director_params(gauss_curv, deform):
    """Calculate parameters for constant curvature spiral angle function

    Args:
        gauss_curv (float): Gaussian curvature of deformed surface
        deform (float): Axial deformation of directors

    Returns:
        tuple: parameters for constant curvature spiral angle function,
               and the maximal allowed radius of the spiral.
    """
    c_k = -gauss_curv / (1 - deform**(-2))
    c = (1 - 2 / (1 + deform))
    rf_max = np.sqrt(2 * (1 - c) / -c_k)
    return c_k, c, rf_max


if __name__ == '__main__':
    num_eval_points = 1000
    ri = MIN_RAD
    connect_len = CONNECTOR_LENGTH
    alpha_func = alpha_const_curvature
    sphere_rad = 190  # mm
    gauss_curv = 1/(sphere_rad**2)
    # deform = 2
    num_connections = 13  # Change this iteratively
    min_length = MIN_LENGTH + num_connections * FRUSTUM_DEFORM
    print(f'{min_length = } mm')
    deform = MAX_LENGTH / min_length
    print(f'Deformation = {deform}')
    args = (gauss_curv, deform)
    _, _, rf_max = const_curvature_director_params(gauss_curv, deform)
    rf = (1 - 1e-6)*rf_max
    num_of_curves = 12
    conj_ratio: int = 2
    # alpha_func = alpha_const
    # alpha = np.pi/2-0.1
    # args = [alpha]
    # rf = 1
    # num_of_curves = 3
    fig, ax, s_events, link_len = compute_axisymmetric_integral_curves(
        alpha_func, ri=ri, rf=rf, num_eval_points=num_eval_points,
        connect_len=connect_len, args=args,
        num_of_curves=num_of_curves,
        conj_ratio=conj_ratio,
        draw_conj=True)
    np.set_printoptions(precision=1)
    print(f'{s_events = }')
    if np.any(link_len):
        print(f'link lengths: {link_len[0]}')
    plt.show(block=True)
