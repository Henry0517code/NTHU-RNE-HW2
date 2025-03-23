import numpy as np


def Bresenham(x0, x1, y0, y1):
    """
    Compute the points of a line between two endpoints using Bresenham's algorithm.

    Args:
        x0 (int): Starting x coordinate.
        x1 (int): Ending x coordinate.
        y0 (int): Starting y coordinate.
        y1 (int): Ending y coordinate.

    Returns:
        list: A list of (x, y) tuples representing the line.
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1

    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    return points


def pos_int(p):
    """
    Convert a coordinate tuple to integers.

    Args:
        p (tuple): A tuple containing coordinate values.

    Returns:
        tuple: A tuple with integer coordinates.
    """
    return (int(p[0]), int(p[1]))


def distance(n1, n2):
    """
    Compute the Euclidean distance (L2 norm) between two points.

    Args:
        n1 (tuple): The first point (x, y).
        n2 (tuple): The second point (x, y).

    Returns:
        float: The Euclidean distance between n1 and n2.
    """
    d = np.array(n1) - np.array(n2)
    return np.hypot(d[0], d[1])