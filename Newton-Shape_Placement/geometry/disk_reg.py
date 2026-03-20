from matplotlib.pylab import outer
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.ops import unary_union
from geometry.regions import all_regions

def create_region(ind=0):
    
    regions = all_regions()
    region = regions[ind]
    return region
 