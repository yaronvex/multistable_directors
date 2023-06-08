import numpy as np
from mayavi import mlab


def prepare_mayavi_multi_line_source(curve_data: list):
    # Curve data: list of numpy arrays, each with shape (3, num_curve_points)
    # Start index of each curve
    curve_len_cumsum = (
        np.cumsum([
            curve_data.shape[-1]
            for curve_data in curve_data])
        - curve_data[0].shape[-1])
    # Line connection arrays
    curve_connections = [
        (np.vstack(
            [np.arange(curve_start_idx,
                       curve_start_idx + curve_data.shape[-1] - 1.5),
             np.arange(curve_start_idx + 1,
                       curve_start_idx + curve_data.shape[-1] - .5)])
         .T)
        for curve_start_idx, curve_data in
        zip(curve_len_cumsum, curve_data)]

    connection_stack = np.vstack(curve_connections)
    curve_stack = np.hstack([curve for curve in curve_data])
    return curve_stack, connection_stack


def init_mayavi_multiline_pipeline(curve_stack, connection_stack):
    curve_src = mlab.pipeline.scalar_scatter(
        curve_stack[0],
        curve_stack[1],
        curve_stack[2])

    # Connect them
    curve_src.mlab_source.dataset.lines = connection_stack
    curve_src.update()

    # The stripper filter cleans up connected lines
    curve_lines = mlab.pipeline.stripper(curve_src)

    return curve_lines
