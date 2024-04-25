"""Module that contains classes describing different boundary conditions
"""

import fipy as fp


def set_dirichlet_boundary_condition(value, faces):
    """ Set the value of a particular concentration variable to a value at the boundary defined by 'faces'

    Args:
        value (float): The scalar value of the concentration variable at the boundary

        faces (boolean numpy.array):  An array containing True/False values about whether that mesh point is on the face

    Returns:
        boundary_value (fp.FixedValue): Object that stores information about the boundary condition
    """

    boundary_value = fp.FixedValue(value=value, faces=faces)
    return boundary_value

