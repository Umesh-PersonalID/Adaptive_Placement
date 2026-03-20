import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union


def build_union(centers, r):
    disks = [Point(c[0], c[1]).buffer(r, quad_segs=80)
             for c in centers]
    return unary_union(disks)

 
def extract_active_segments(centers, r, region):

    m = len(centers)
    union = build_union(centers, r)

    boundary = union.intersection(region).boundary

    active_segments = [[] for _ in range(m)]

    # Handle single or multi line
    if boundary.geom_type == "LineString":
        segments = [boundary]
    else:
        segments = list(boundary.geoms)

    for seg in segments:

        coords = list(seg.coords)

        for i in range(len(coords) - 1):

            p = np.array(coords[i])
            q = np.array(coords[i + 1])

            mid = 0.5 * (p + q)
            for disk_id, c in enumerate(centers):

                if abs(np.linalg.norm(mid - c) - r) < 1e-2:
                    n_hat = (mid - c) / r
                    active_segments[disk_id].append({
                        "p": p,
                        "q": q,
                        "mid": mid,
                        "normal": n_hat
                    })
                    break
    return active_segments


def grad_G(x, r, region):

    m = len(x) // 2
    centers = x.reshape(m, 2)

    active_segments = extract_active_segments(centers, r, region)

    grad = np.zeros_like(x)

    for i in range(m):
        g_i = np.zeros(2)
        for seg in active_segments[i]:
            n = seg["normal"]
            ds = np.linalg.norm(seg["q"] - seg["p"])
            g_i += n * ds
        grad[2*i:2*i+2] = g_i
    return -1*grad