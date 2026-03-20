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
        (0, 0), (30, 0), (30, 15),
        (25, 20), (15, 25),
        (0, 20)
    ] 

    hole4_1 = [(10, 5), (14, 5), (14, 10), (10, 10)]
    hole4_2 = [(20, 7), (24, 7), (24, 12), (20, 12)]
    hole4_3 = [(12, 18), (15, 22), (18, 18), (14,10), (12, 18)]
    hole4_4 = Point(22, 12).buffer(2, resolution=64)  
    hole4_5 = Point(22, 7).buffer(2, resolution=64)
    hole4_6 = [(10, 10), (14,10), (12,18) ,(10, 15), (10, 10)]
    hole4_7 = Point(10, 10).buffer(3, resolution=64)

    region4 = Polygon(outer4, [hole4_1, hole4_2, hole4_3])
    region4 = region4.difference(hole4_4)
    region4 = region4.difference(hole4_5)
    region4 = region4.difference(Polygon(hole4_6))
    region4 = region4.difference(hole4_7)
    regions.append(region4)

    outer5 = [
        (0, 10), (5, 20), (25, 20),
        (30, 10), (20, 10),
        (20, 0), (10, 0),
        (10, 10)
    ]

    hole5_1 = [(12, 12), (16, 12), (16, 16), (12, 16)]
    hole5_2 = [(14, 4), (18, 4), (18, 12), (14, 11.9999999)]

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

    hole1 = Point(10, 7).buffer(3.5, resolution=32)
    hole2 = box(24, 5, 30, 12)

    hole3 = Point(27, 12).buffer(3, resolution=24)
    hole4 = rotate(box(8, 3, 14, 7), 20, origin='center')

    curved_region = curved_outer.difference(
        unary_union([hole1, hole2, hole3, hole4])
    )

    regions.append(curved_region)


    base = box(0, 0, 40, 20)
    dome = Point(12, 20).buffer(10, resolution=64)
    dome_clip = box(-20, 20, 40, 50)
    curved_top = dome.intersection(dome_clip)

    outer = unary_union([base, curved_top])

    notch = box(15, 0, 25, 8)
    outer = outer.difference(notch)

    hole1 = Point(20, 8).buffer(3, resolution=48)
    rect_hole = box(30, 8, 36, 14)
    hole2 = rotate(rect_hole, 25, origin='center')

    hole3 = Point(12, 20).buffer(6, resolution=32)
    hole4 = [(42, 8), (48, 12), (45, 16)]  

    final_region = outer.difference(
        unary_union([hole1, hole2, hole3, Polygon(hole4)])
    )

    regions.append(final_region)

    outer6 = [
        (0, 0), (32, 0), (32, 8),
        (26, 8), (26, 16),
        (18, 20), (8, 16),
        (0, 10)
    ]

    hole6_1 = [(6, 4), (10, 4), (10, 8), (6, 8)]
    hole6_2 = box(20, 3, 25, 7)

    circle_cut = Point(14, 12).buffer(2.5, resolution=32)
    rotated_cut = rotate(box(22, 12, 28, 16), 35, origin='center')

    region6 = Polygon(outer6, [hole6_1])
    region6 = region6.difference(unary_union([hole6_2, circle_cut, rotated_cut]))

    regions.append(region6)


    base7 = box(0, 0, 36, 16)
    crown = Point(18, 16).buffer(10, resolution=64)
    crown_clip = box(-10, 16, 50, 40)
    top_curve = crown.intersection(crown_clip)

    outer7 = unary_union([base7, top_curve])

    notch7 = Polygon([(14, 0), (22, 0), (18, 6)])
    outer7 = outer7.difference(notch7)

    hole7_2 = rotate(box(24, 6, 30, 12), 20, origin='center')
    hole7_3 = [(5, 5), (5, 14), (15, 14), (11, 5), (5, 5)]

    region7 = outer7.difference(
        unary_union([hole7_2, Polygon(hole7_3)])
    )

    regions.append(region7)


    return regions

if __name__ == "__main__":
    # Example usage: Print the number of regions and their areas
    regions = all_regions()
    for i, region in enumerate(regions):
        print(f"Region {i+1}: Area = {region.area:.2f}")