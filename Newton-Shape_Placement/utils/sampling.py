import numpy as np
from shapely.geometry import Point
from shapely.ops import nearest_points

def sample_points_in_polygon(polygon, n, max_trials_per_point=1000):
    minx, miny, maxx, maxy = polygon.bounds
    points = []

    max_trials = n * max_trials_per_point
    trials = 0

    while len(points) < n:
        if trials >= max_trials:
            raise RuntimeError(
                f"Sampling failed after {max_trials} trials. "
                f"Only obtained {len(points)} of {n} points."
            )

        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)

        if polygon.contains(Point(x, y)):
            points.append((x, y))

        trials += 1

    return np.array(points)




def project_points_to_polygon(points, polygon, eps=1e-6):
    projected = np.array(points, dtype=float).copy()

    for i, (x, y) in enumerate(projected):
        p = Point(float(x), float(y))

        if polygon.contains(p):
            continue

        _, nearest_on_poly = nearest_points(p, polygon)
        nx, ny = nearest_on_poly.x, nearest_on_poly.y
        q = np.array([nx, ny], dtype=float)
        c = np.array([x, y], dtype=float)

        direction = q - c
        norm = np.linalg.norm(direction)

        if norm > 1e-12:
            q = q + eps * (direction / norm)

        projected[i] = q

    return projected
