from shapely.geometry import Polygon
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
from shapely.affinity import rotate

from shapely.geometry import Polygon, Point, box
from shapely.affinity import rotate
from shapely.ops import unary_union


def all_regions():
    regions = []
    outer4 = [
        (5, 0), (30, 0), (30, 15),
        (25, 20), (15, 25),
        (5, 20)
    ]

    hole4_1 = [(10, 5), (14, 5), (14, 10), (10, 10)]
    hole4_2 = [(20, 7), (24, 7), (24, 12), (20, 12)]
    hole4_3 = [(12, 18), (15, 22), (18, 18)]  
    hole4_4 = Point(22, 16).buffer(2, resolution=32)  

    region4 = Polygon(outer4, [hole4_1, hole4_2, hole4_3])
    region4 = region4.difference(hole4_4)

    regions.append(region4)

    outer5 = [
        (0, 10), (5, 20), (25, 20),
        (30, 10), (20, 10),
        (20, 0), (10, 0),
        (10, 10)
    ]

    hole5_1 = [(12, 12), (16, 12), (16, 16), (12, 16)]
    hole5_2 = [(14, 4), (16, 4), (16, 8), (14, 8)]

    hole5_3 = Point(8, 15).buffer(1.8, resolution=24)
    tilted = rotate(box(21, 13, 26, 17), 30, origin='center')

    region5 = Polygon(outer5, [hole5_1, hole5_2])
    region5 = region5.difference(unary_union([hole5_3, tilted]))

    regions.append(region5)
    base = box(0, 0, 40, 20)
    dome = Point(20, 20).buffer(12, resolution=64)
    clip_box = box(-10, 20, 50, 50)
    half_dome = dome.intersection(clip_box)

    curved_outer = unary_union([base, half_dome])

    hole1 = Point(12, 10).buffer(3, resolution=32)
    hole2 = box(24, 5, 30, 12)

    hole3 = Point(30, 15).buffer(2, resolution=24)
    hole4 = rotate(box(8, 3, 14, 7), 20, origin='center')

    curved_region = curved_outer.difference(
        unary_union([hole1, hole2, hole3, hole4])
    )

    regions.append(curved_region)
    base = box(0, 0, 40, 20)
    dome = Point(12, 20).buffer(10, resolution=64)
    dome_clip = box(-20, 20, 40, 50)
    curved_top = dome.intersection(dome_clip)

    extension = box(40, 5, 55, 18)
    outer = unary_union([base, curved_top, extension])

    notch = box(15, 0, 25, 8)
    outer = outer.difference(notch)

    hole1 = Point(18, 12).buffer(3, resolution=48)
    rect_hole = box(30, 8, 36, 14)
    hole2 = rotate(rect_hole, 25, origin='center')

    hole3 = Point(8, 8).buffer(2, resolution=32)
    hole4 = [(42, 8), (48, 12), (45, 16)]  

    final_region = outer.difference(
        unary_union([hole1, hole2, hole3, Polygon(hole4)])
    )

    regions.append(final_region)

    return regions