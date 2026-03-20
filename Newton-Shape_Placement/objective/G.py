import numpy as np
from objective.gradient import build_union

def G(x, r, region):
    m = len(x) // 2
    centers = x.reshape(m, 2)
    omega = build_union(centers, r)
    covered = region.intersection(omega).area
    return region.area - covered

if __name__ == "__main__":
    pass 