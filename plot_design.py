import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, MultiPolygon, LineString, GeometryCollection
from shapely.validation import make_valid
from shapely.ops import unary_union, polygonize
from shapely import affinity
from concurrent.futures import ThreadPoolExecutor
import time

deg = np.pi / 180

def filter_to_polygons(geom):
    """Returns a Polygon, MultiPolygon, or empty Polygon from any geometry."""
    if geom.is_empty:
        return Polygon()

    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom

    if isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
        if not polys:
            return Polygon()
        elif len(polys) == 1:
            return polys[0]
        else:
            return MultiPolygon(polys)

    # All other cases (LineString, Point, etc.)
    return Polygon()


def plot_polygons(geom, title="Polygon(s)"):
    fig, ax = plt.subplots()
    if geom.geom_type == 'Polygon':
        x, y = geom.exterior.xy
        ax.plot(x, y, color='blue')
    elif geom.geom_type == 'MultiPolygon':
        for poly in geom.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, color='green')
    else:
        print(f"Unsupported geometry type: {geom.geom_type}")
    ax.set_title(title)
    ax.set_aspect('equal')
    plt.show()


def plot_filled_polygons(geom, title="Filled Polygon(s)", facecolor='skyblue', edgecolor='black'):
    fig, ax = plt.subplots()
    patches = []

    def add_polygon(p):
        # Exterior ring
        patches.append(MplPolygon(list(p.exterior.coords), closed=True))
        # Interior rings (holes)
        for interior in p.interiors:
            patches.append(MplPolygon(list(interior.coords), closed=True, fill=True))

    if isinstance(geom, Polygon):
        add_polygon(geom)
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            add_polygon(poly)
    else:
        print("Unsupported geometry type:", geom.geom_type)
        return

    p_collection = PatchCollection(patches, facecolor=facecolor, edgecolor=edgecolor, linewidth=1)
    ax.add_collection(p_collection)
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_title(title)
    plt.show()


def overlap_FOV(polygon1, polygon2):
    """
    Returns:
    --------
    overlap_region: Polygon or MultiPolygon or empty Polygon
        The intersection region of the two input polygons.
    modified_polygon: Polygon or MultiPolygon or empty Polygon
        The second polygon with the overlap region removed.
    """
    if not polygon1.is_valid:
        polygon1 = make_valid(polygon1)
        polygon1 = filter_to_polygons(polygon1)
    if not polygon2.is_valid:
        polygon2 = make_valid(polygon2)
        polygon2 = filter_to_polygons(polygon2)

    # Compute and sanitize the overlap region
    raw_overlap = polygon1.intersection(polygon2)
    overlap_region = filter_to_polygons(raw_overlap)

    # Compute and sanitize the modified polygon
    if overlap_region.is_empty:
        modified_polygon = polygon2
    else:
        raw_modified = polygon2.difference(overlap_region)
        modified_polygon = filter_to_polygons(raw_modified)

    return overlap_region, modified_polygon


def polygon_to_xy(polygon, xs_list, ys_list):
    """
    Converts a Polygon or MultiPolygon to x, y coordinate arrays
    and appends them to xs_list and ys_list.

    Parameters:
        polygon (Polygon or MultiPolygon): The input geometry.
        xs_list (list): List to append x-coordinates (as arrays).
        ys_list (list): List to append y-coordinates (as arrays).
    """
    if polygon.is_empty:
        return

    if isinstance(polygon, Polygon):
        x, y = polygon.exterior.xy
        xs_list.append(x)  # x is already an array
        ys_list.append(y)

    elif isinstance(polygon, MultiPolygon):
        for poly in polygon.geoms:
            if not poly.is_empty:
                x, y = poly.exterior.xy
                xs_list.append(x)
                ys_list.append(y)


def count_polygons(geometry):
    if geometry.is_empty:
        return 0
    elif isinstance(geometry, Polygon):
        return 1
    elif isinstance(geometry, MultiPolygon):
        return len(geometry.geoms)
    else:
        raise TypeError("Input is not a Polygon or MultiPolygon.")


def poly2mat(polygon_input, eff_saved, x_region, y_region, resolution):
    # Ensure the input is Polygon or MultiPolygon
    if isinstance(polygon_input, (Polygon, MultiPolygon)):
        polygons = polygon_input.geoms if isinstance(polygon_input, MultiPolygon) else [polygon_input]
    else:
        # Assume it's a sequence of (x, y) points
        polygons = [Polygon(polygon_input)]

    # Bounding box
    x_max = np.max(x_region)
    x_min = np.min(x_region)
    y_max = np.max(y_region)
    y_min = np.min(y_region)

    # Create meshgrid
    x, y = np.meshgrid(
        np.arange(x_min, x_max + resolution, resolution),
        np.arange(y_min, y_max + resolution, resolution)
    )

    # Flatten grid
    points = np.vstack((x.flatten(), y.flatten())).T

    # Initialize combined mask
    in_poly_grid = np.zeros_like(x, dtype=bool)

    # Check each polygon part
    for poly in polygons:
        path = Path(np.array(poly.exterior.coords))
        in_poly = path.contains_points(points).reshape(x.shape)
        in_poly_grid |= in_poly  # logical OR to accumulate coverage

    # Multiply by efficiency
    matrix = in_poly_grid.astype(float) * eff_saved

    return matrix

"""## Design goal of waveguide"""

# choose the optimize FOV number
FOV_num = 1

# Field of View (FoV)
AR = 4 / 3  # aspect ratio
FoV_x = 20 * deg  # AR=4/3; FoV_x=40*deg;
FoV_y = FoV_x / AR
FoV_X = np.linspace(-FoV_x / 2, FoV_x / 2, 50)  # coordinates for each FoV point
FoV_Y = np.linspace(-FoV_y / 2, FoV_y / 2, 50)

# Wavelength
lmd = 532
k0 = 2 * np.pi / lmd

# Substrate parameters
n_g = 1.52  # index of glass
n_air = 1
x = 60
y = 50  # glass size
t = 0.7  # thickness of the waveguide substrate

# Input pupil size (circular) in mm
r = 2  # radius

# Input pupil position
x_ic0 = -28  # coordinates of center of input coupler/pupil
y_ic0 = 15
n = 100  # number of points
t_ic = np.linspace(0, 2 * np.pi, n)
X_ic = x_ic0 + r * np.sin(t_ic)
Y_ic = y_ic0 + r * np.cos(t_ic)

# Eyebox size (mm)
x_eb = 12
y_eb = 8
er = -20  # same side or different side for LE and HE have same folding

# Eyebox position (mm)
x_eb0 = 0  # x_eb0 = 13
y_eb0 = 15
X_eb = np.array([-x_eb / 2, -x_eb / 2, x_eb / 2, x_eb / 2]) + x_eb0
Y_eb = np.array([-y_eb / 2, y_eb / 2, y_eb / 2, -y_eb / 2]) + y_eb0

# Out-coupler size (mm)
x_oc = np.tan(FoV_x / 2) * abs(er) * 2 + x_eb
y_oc = np.tan(FoV_y / 2) * abs(er) * 2 + y_eb

# Out-coupler position
X_oc = np.array([-x_oc / 2, -x_oc / 2, x_oc / 2, x_oc / 2]) + x_eb0
Y_oc = np.array([-y_oc / 2, y_oc / 2, y_oc / 2, -y_oc / 2]) + y_eb0

# Maximum and minimum TIR angle for design
theta_min = np.arcsin(n_air / n_g)  # minimum TIR angle
theta_max = np.arctan(r / t)        # maximum TIR angle if pupil is continuous at largest TIR angle

"""## Checking periods of input and output coupler for designed FoV"""

# Horizontal period and k_g direction of input coupler
Lambda_ic = 437
phi_ic = -38 * deg

# Horizontal period and k_g direction of output coupler
Lambda_oc = 437
phi_oc = -142 * deg

# Incoupler k vector
kg_ic = 2 * np.pi / Lambda_ic
kgx_ic = kg_ic * np.cos(phi_ic)
kgy_ic = kg_ic * np.sin(phi_ic)

# Reverse direction of outcoupler k vector
kg_oc = 2 * np.pi / Lambda_oc
kgx_oc = kg_oc * np.cos(phi_oc + 180 * deg)
kgy_oc = kg_oc * np.sin(phi_oc + 180 * deg)

"""## Calculating period and shape of folding couplers"""

# k-vector and horizontal period of folding coupler
kgx_fc = kgx_oc - kgx_ic
kgy_fc = kgy_oc - kgy_ic
Lambda_fc = 2 * np.pi / np.sqrt(kgx_fc**2 + kgy_fc**2)
print(Lambda_fc)
phi_fc = np.degrees(np.arctan2(kgy_fc, kgx_fc))  # use arctan2 for stability

kk = 0

# Preallocate result arrays
kx0 = np.zeros((1, len(FoV_X)*len(FoV_Y)))
ky0 = np.zeros((1, len(FoV_X)*len(FoV_Y)))
kx_ic = np.zeros_like(kx0)
ky_ic = np.zeros_like(ky0)
kx_fc = np.zeros_like(kx0)
ky_fc = np.zeros_like(ky0)
x_f = []
y_f = []

for ii in range(len(FoV_X)):
    for jj in range(len(FoV_Y)):
        # k-vector in air
        th_inc = np.arctan(np.sqrt(np.tan(FoV_X[ii])**2 + np.tan(FoV_Y[jj])**2))
        phi_inc = np.arctan2(np.tan(FoV_Y[jj]), np.tan(FoV_X[ii]))
        kx0[0,kk] = n_air * k0 * np.sin(th_inc) * np.cos(phi_inc)
        ky0[0,kk] = n_air * k0 * np.sin(th_inc) * np.sin(phi_inc)

        # k-vector after incoupler
        kx_ic[0,kk] = kx0[0,kk] + kgx_ic
        ky_ic[0,kk] = ky0[0,kk] + kgy_ic
        kz_ic = np.sqrt(k0**2 * n_g**2 - kx_ic[0,kk]**2 - ky_ic[0,kk]**2)

        # tangent lines of input pupil for different FoV after input coupler
        k1 = ky_ic[0,kk] / kx_ic[0,kk]
        b11 = y_ic0 - k1 * x_ic0 + r * np.sqrt(1 + k1**2)
        b12 = y_ic0 - k1 * x_ic0 - r * np.sqrt(1 + k1**2)

        # k-vector after folding
        kx_fc[0,kk] = kx_ic[0,kk] + kgx_fc
        ky_fc[0,kk] = ky_ic[0,kk] + kgy_fc
        kz_fc = np.sqrt(k0**2 * n_g**2 - kx_fc[0,kk]**2 - ky_fc[0,kk]**2)

        # position of each edge
        dx = er * np.tan(th_inc) * np.cos(phi_inc)
        dy = er * np.tan(th_inc) * np.sin(phi_inc)

        x_ed_left_t = x_eb0 - x_eb / 2 + dx
        y_ed_left_t = y_eb0 + y_eb / 2 + dy
        x_ed_right_b = x_eb0 + x_eb / 2 + dx
        y_ed_right_b = y_eb0 - y_eb / 2 + dy
        x_ed_left_b = x_eb0 - x_eb / 2 + dx
        y_ed_left_b = y_eb0 - y_eb / 2 + dy
        x_ed_right_t = x_eb0 + x_eb / 2 + dx
        y_ed_right_t = y_eb0 + y_eb / 2 + dy

        # tangent lines of output coupler for different FoV after folding coupler
        k2 = ky_fc[0,kk] / kx_fc[0,kk]
        if k2 <= 0:
            b21 = y_ed_left_b - k2 * x_ed_left_b
            b22 = y_ed_right_t - k2 * x_ed_right_t
        else:
            b21 = y_ed_left_t - k2 * x_ed_left_t
            b22 = y_ed_right_b - k2 * x_ed_right_b

        # intersection points of two lines (x = (b2 - b1)/(k1 - k2))
        for b1 in [b11, b12]:
            for b2 in [b22, b21]:
                x_inter = (b2 - b1) / (k1 - k2)
                y_inter = k1 * x_inter + b1
                x_f.append(x_inter)
                y_f.append(y_inter)
        kk += 1

"""## Plot k diagram"""

fig = plt.figure()
c = [0, 0]
r1 = n_air
r2 = n_g
r3 = n_g * np.sin(theta_max)

# Circle coordinates
x1 = c[0] + r1 * np.sin(t_ic)
y1 = c[1] + r1 * np.cos(t_ic)
x2 = c[0] + r2 * np.sin(t_ic)
y2 = c[1] + r2 * np.cos(t_ic)
x3 = c[0] + r3 * np.sin(t_ic)
y3 = c[1] + r3 * np.cos(t_ic)

# Draw k-diagram boundaries
plt.plot(x1, y1, label='Air boundary')
plt.plot(x2, y2, label='Glass boundary')
plt.plot(x3, y3, '--', label='Max TIR angle')

# Combine into a single (N, 2) array of points
points0 = np.vstack((kx0 / k0, ky0 / k0))
points_ic = np.vstack((kx_ic / k0, ky_ic / k0))
points_fc = np.vstack((kx_fc / k0, ky_fc / k0))

# Compute convex hull
hull0 = ConvexHull(np.transpose(points0))
hull_ic = ConvexHull(np.transpose(points_ic))
hull_fc = ConvexHull(np.transpose(points_fc))

# Fill the convex hull polygon
plt.fill(points0[0, hull0.vertices], points0[1, hull0.vertices], 'blue', alpha=0.5)
plt.fill(points_ic[0, hull_ic.vertices], points_ic[1, hull_ic.vertices], 'red', alpha=0.5)
plt.fill(points_fc[0, hull_fc.vertices], points_fc[1, hull_fc.vertices], 'green', alpha=0.5)

plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.grid(False)
plt.title('k-diagram')
plt.xlabel('kx/k0')
plt.ylabel('ky/k0')
plt.show()

"""## Generate 9 FoV for optimization"""

# Generate 9 FoVs
FoV_X_9c = np.array([
    -FoV_x / 2, np.finfo(float).eps, FoV_x / 2,
    -FoV_x / 2, np.finfo(float).eps, FoV_x / 2,
     FoV_x / 2, np.finfo(float).eps, -FoV_x / 2
])
FoV_Y_9c = np.array([
     FoV_y / 2, FoV_y / 2, FoV_y / 2,
     np.finfo(float).eps, np.finfo(float).eps, np.finfo(float).eps,
    -FoV_y / 2, -FoV_y / 2, -FoV_y / 2
])

# Compute convex hull of the folded coupling region
points_fc = np.vstack((x_f, y_f)).T
hull_fc = ConvexHull(points_fc)
bd = hull_fc.vertices

plt.figure()
plt.fill(X_ic, Y_ic, 'r', edgecolor='black')
plt.axis('equal')
plt.xlim([-x / 2, x / 2])
plt.ylim([-y / 2, y / 2])

# Rotate the folding region
points = np.vstack((x_f, y_f))[:, bd]
angle = np.pi / 2 + phi_ic
rotation_2d = np.array([[np.cos(angle), np.sin(angle)],
             [-np.sin(angle), np.cos(angle)]])
rotated_points = rotation_2d @ points

# Define slicing region
start_line = np.max(rotated_points[1, :])
end_line = np.min(rotated_points[1, :])
slice_width = (start_line - end_line)/7.001
num_slices = int(np.ceil((start_line - end_line) / slice_width))
end_width = (start_line - end_line) % slice_width
if end_width < slice_width / 4:
    num_slices -= 1

# Preallocate arrays for FoV shape
x_fc_FOV = np.zeros((len(FoV_Y_9c), 4))
y_fc_FOV = np.zeros((len(FoV_Y_9c), 4))
kx0_9c = np.zeros(len(FoV_Y_9c))
ky0_9c = np.zeros(len(FoV_Y_9c))

# Compute shape for each 9-cell FoV region
for ii in range(len(FoV_Y_9c)):
    th_inc = np.arctan(np.sqrt(np.tan(FoV_X_9c[ii])**2 + np.tan(FoV_Y_9c[ii])**2))
    phi_inc = np.arctan2(np.tan(FoV_Y_9c[ii]), np.tan(FoV_X_9c[ii]))

    kx0_9c[ii] = n_air * k0 * np.sin(th_inc) * np.cos(phi_inc)
    ky0_9c[ii] = n_air * k0 * np.sin(th_inc) * np.sin(phi_inc)

    kx_ic_ = kx0_9c[ii] + kgx_ic
    ky_ic_ = ky0_9c[ii] + kgy_ic
    kz_ic = np.sqrt(k0**2 * n_g**2 - kx_ic_**2 - ky_ic_**2)

    k1 = ky_ic_ / kx_ic_
    b11 = y_ic0 - k1 * x_ic0 + r * np.sqrt(1 + k1**2)
    b12 = y_ic0 - k1 * x_ic0 - r * np.sqrt(1 + k1**2)

    kx_fc_ = kx_ic_ + kgx_fc
    ky_fc_ = ky_ic_ + kgy_fc
    kz_fc = np.sqrt(k0**2 * n_g**2 - kx_fc_**2 - ky_fc_**2)

    dx = er * np.tan(th_inc) * np.cos(phi_inc)
    dy = er * np.tan(th_inc) * np.sin(phi_inc)

    x_ed_l_t = x_eb0 - x_eb / 2 + dx
    y_ed_l_t = y_eb0 + y_eb / 2 + dy
    x_ed_r_b = x_eb0 + x_eb / 2 + dx
    y_ed_r_b = y_eb0 - y_eb / 2 + dy
    x_ed_l_b = x_eb0 - x_eb / 2 + dx
    y_ed_l_b = y_eb0 - y_eb / 2 + dy
    x_ed_r_t = x_eb0 + x_eb / 2 + dx
    y_ed_r_t = y_eb0 + y_eb / 2 + dy

    k2 = ky_fc_ / kx_fc_
    if k2 <= 0:
        b21 = y_ed_l_b - k2 * x_ed_l_b
        b22 = y_ed_r_t - k2 * x_ed_r_t
    else:
        b21 = y_ed_l_t - k2 * x_ed_l_t
        b22 = y_ed_r_b - k2 * x_ed_r_b

    x_fc_FOV[ii, 0] = (b22 - b11) / (k1 - k2)
    x_fc_FOV[ii, 1] = (b21 - b11) / (k1 - k2)
    x_fc_FOV[ii, 2] = (b21 - b12) / (k1 - k2)
    x_fc_FOV[ii, 3] = (b22 - b12) / (k1 - k2)

    y_fc_FOV[ii, 0] = k1 * x_fc_FOV[ii, 0] + b11
    y_fc_FOV[ii, 1] = k1 * x_fc_FOV[ii, 1] + b11
    y_fc_FOV[ii, 2] = k1 * x_fc_FOV[ii, 2] + b12
    y_fc_FOV[ii, 3] = k1 * x_fc_FOV[ii, 3] + b12

# Store valid slice polygons
points_x_FOV_fc = []
points_y_FOV_fc = []
kk = 0
FOV_num -= 1
poly_FOV_fc = Polygon(np.column_stack((x_fc_FOV[FOV_num, :], y_fc_FOV[FOV_num, :])))
poly_FOV_fc = make_valid(poly_FOV_fc)

for i in range(1, num_slices + 1):
    # Current slice y-bounds in rotated coordinates
    col_start = start_line - (i - 1) * slice_width
    col_end = start_line - i * slice_width

    points_x = rotated_points[0, :]
    points_y = rotated_points[1, :]

    # Polygon of the current folded region
    poly_fc = Polygon(np.column_stack((points_x, points_y)))

    # Clip polygon between the top and bottom lines
    if i == num_slices:
        band = Polygon([
            (-x, col_start), (x, col_start),
            (x, end_line), (-x, end_line)
        ])
    else:
        band = Polygon([
            (-x, col_start), (x, col_start),
            (x, col_end), (-x, col_end)
        ])

    clipped = poly_fc.intersection(band)
    if clipped.is_empty:
        continue

    # Convert back to original coordinates
    if clipped.geom_type == 'Polygon':
        coords = np.array(clipped.exterior.coords).T
    elif clipped.geom_type == 'MultiPolygon':
        coords = np.array(list(clipped.geoms[0].exterior.coords)).T
    else:
        continue

    # Rotate back
    inv_rotation = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    restored = inv_rotation @ coords
    px_restored, py_restored = restored[0, :], restored[1, :]
    p_restored_poly = Polygon(np.column_stack((px_restored, py_restored)))
    p_restored_poly = make_valid(p_restored_poly)

    # Check overlap with center FoV polygon
    overlapRegion, modifiedPolygon = overlap_FOV(p_restored_poly, poly_FOV_fc)

    if not overlapRegion.is_empty:
        kk += 1
        plt.fill(px_restored, py_restored, 'g', alpha=0.6, edgecolor='black')
        points_x_FOV_fc.append(px_restored)
        points_y_FOV_fc.append(py_restored)
    else:
        plt.fill(px_restored, py_restored, 'g', alpha=0.3, edgecolor='black')

num_FOV_fc = kk

# Combine all folded-coupler subregions into one
if num_FOV_fc > 0:
    points_x_FOV_fc_all = np.concatenate(points_x_FOV_fc)
    points_y_FOV_fc_all = np.concatenate(points_y_FOV_fc)
    all_fc_points = np.column_stack((points_x_FOV_fc_all, points_y_FOV_fc_all))
    bd = ConvexHull(all_fc_points).vertices
    points_x_FOV_fc_all = points_x_FOV_fc_all[bd]
    points_y_FOV_fc_all = points_y_FOV_fc_all[bd]
    # Optional: visualize
    # plt.fill(points_x_FOV_fc_all, points_y_FOV_fc_all, 'g', alpha=0.3)

# === Rotate out-coupler polygon ===
points_oc = np.vstack((X_oc, Y_oc)).T
hull_oc = ConvexHull(points_oc)
bd = hull_oc.vertices
points = np.vstack((X_oc[bd], Y_oc[bd]))

# Rotate the out-coupler region
angle_oc = 3 * np.pi / 2 + phi_oc
rotation_2d_oc = np.array([
    [np.cos(angle_oc), np.sin(angle_oc)],
    [-np.sin(angle_oc), np.cos(angle_oc)]
])
rotated_oc = rotation_2d_oc @ points

# Define slicing parameters
start_line = np.max(rotated_oc[1, :])
end_line = np.min(rotated_oc[1, :])
slice_width = (start_line - end_line)/6.001
num_slices = int(np.ceil((start_line - end_line) / slice_width))
end_width = (start_line - end_line) % slice_width
if end_width < slice_width / 4:
    num_slices -= 1

# Effective out-coupler region polygons
x_oc_FOV = np.zeros((9, 4))
y_oc_FOV = np.zeros((9, 4))
for j in range(9):
    th_inc = np.arctan(np.sqrt(np.tan(FoV_X_9c[j])**2 + np.tan(FoV_Y_9c[j])**2))
    phi_inc = np.arctan2(np.tan(FoV_Y_9c[j]), np.tan(FoV_X_9c[j]))

    dx = er * np.tan(th_inc) * np.cos(phi_inc)
    dy = er * np.tan(th_inc) * np.sin(phi_inc)

    x_oc_FOV[j, :] = np.array([
        x_eb0 - x_eb/2 + dx,
        x_eb0 - x_eb/2 + dx,
        x_eb0 + x_eb/2 + dx,
        x_eb0 + x_eb/2 + dx
    ])

    y_oc_FOV[j, :] = np.array([
        y_eb0 + y_eb/2 + dy,
        y_eb0 - y_eb/2 + dy,
        y_eb0 - y_eb/2 + dy,
        y_eb0 + y_eb/2 + dy
    ])

# Optional visualization
plt.fill(x_oc_FOV[FOV_num, :], y_oc_FOV[FOV_num, :], 'r', alpha=0.3)
plt.fill(x_fc_FOV[FOV_num, :], y_fc_FOV[FOV_num, :], 'r', alpha=0.3)

# Highlight combined region
effective_x = np.concatenate([x_oc_FOV[FOV_num, :], x_fc_FOV[FOV_num, :]])
effective_y = np.concatenate([y_oc_FOV[FOV_num, :], y_fc_FOV[FOV_num, :]])
bd2 = ConvexHull(np.column_stack((effective_x, effective_y))).vertices
effective_x = effective_x[bd2]
effective_y = effective_y[bd2]
plt.fill(effective_x, effective_y, 'r', alpha=0.3)
poly_FOV_oc = Polygon(np.column_stack((x_oc_FOV[FOV_num, :], y_oc_FOV[FOV_num, :])))

# === Slice and intersect out-coupler ===
points_x_FOV_oc = []
points_y_FOV_oc = []
kk = 0

for i in range(1, num_slices + 1):
    col_start = start_line - (i - 1) * slice_width
    col_end = start_line - i * slice_width

    px = rotated_oc[0, :]
    py = rotated_oc[1, :]

    # Clip polygon between the top and bottom lines
    if i == num_slices:
        slice_band = Polygon([
            (-x, col_start), (x, col_start),
            (x, end_line), (-x, end_line)
        ])
    else:
        slice_band = Polygon([
            (-x, col_start), (x, col_start),
            (x, col_end), (-x, col_end)
        ])

    poly_oc = Polygon(np.column_stack((px, py)))
    clipped = poly_oc.intersection(slice_band)
    if clipped.is_empty:
        continue

    # Handle multi-polygon case
    if clipped.geom_type == 'Polygon':
        coords = np.array(clipped.exterior.coords).T
    elif clipped.geom_type == 'MultiPolygon':
        coords = np.array(list(clipped.geoms[0].exterior.coords)).T
    else:
        continue

    # Rotate back
    inv_rotation_oc = np.array([
        [np.cos(angle_oc), -np.sin(angle_oc)],
        [np.sin(angle_oc),  np.cos(angle_oc)]
    ])
    restored = inv_rotation_oc @ coords
    px_restored, py_restored = restored[0, :], restored[1, :]
    p_restored_poly = Polygon(np.column_stack((px_restored, py_restored)))

    # Test overlap with output coupler FOV region
    overlapRegion, modifiedPolygon = overlap_FOV(p_restored_poly, poly_FOV_oc)

    if not overlapRegion.is_empty:
        kk += 1
        plt.fill(px_restored, py_restored, 'b', alpha=0.6, edgecolor='black')
        points_x_FOV_oc.append(px_restored)
        points_y_FOV_oc.append(py_restored)
    else:
        plt.fill(px_restored, py_restored, 'b', alpha=0.3, edgecolor='black')


num_FOV_oc = kk

# Combine all oc region into one region
if num_FOV_oc > 0:
    all_x = np.concatenate(points_x_FOV_oc)
    all_y = np.concatenate(points_y_FOV_oc)
    bd = ConvexHull(np.column_stack((all_x, all_y))).vertices
    points_x_FOV_oc_all = all_x[bd]
    points_y_FOV_oc_all = all_y[bd]
    # Optional plot
    # plt.fill(points_x_FOV_oc_all, points_y_FOV_oc_all, 'b', alpha=0.3)

th_out_ic = np.zeros(9)
gap_x_ic = np.zeros(9)
gap_y_ic = np.zeros(9)
th_out_fc = np.zeros(9)
gap_x_fc = np.zeros(9)
gap_y_fc = np.zeros(9)

for i in range(9):
    th = np.arctan(np.sqrt(np.tan(FoV_X_9c[i])**2 + np.tan(FoV_Y_9c[i])**2))
    phi = np.arctan2(np.tan(FoV_Y_9c[i]), np.tan(FoV_X_9c[i]))

    # k-vector in air
    kx = n_air * k0 * np.sin(th) * np.cos(phi)
    ky = n_air * k0 * np.sin(th) * np.sin(phi)

    # after input coupler
    kxg_ic = kx + kgx_ic
    kyg_ic = ky + kgy_ic
    kzg_ic = np.sqrt(k0**2 * n_g**2 - kxg_ic**2 - kyg_ic**2)
    th_out_ic[i] = np.arctan(np.sqrt((kxg_ic**2 + kyg_ic**2) / kzg_ic**2)) / deg

    k1 = kyg_ic / kxg_ic
    gap_x_ic[i] = 2 * t * np.tan(th_out_ic[i] * deg) * np.cos(np.arctan(k1))
    gap_y_ic[i] = 2 * t * np.tan(th_out_ic[i] * deg) * np.sin(np.arctan(k1))

    # after folded coupler
    kxg_fc = kxg_ic + kgx_fc
    kyg_fc = kyg_ic + kgy_fc
    kzg_fc = np.sqrt(k0**2 * n_g**2 - kxg_fc**2 - kyg_fc**2)
    th_out_fc[i] = np.arctan(np.sqrt((kxg_fc**2 + kyg_fc**2) / kzg_fc**2)) / deg

    k2 = kyg_fc / kxg_fc
    gap_x_fc[i] = 2 * t * np.tan(th_out_fc[i] * deg) * np.cos(np.arctan(k2))
    gap_y_fc[i] = 2 * t * np.tan(th_out_fc[i] * deg) * np.sin(np.arctan(k2))

# === Region cut at input coupler ===
X_ic2 = X_ic + gap_x_ic[FOV_num]
Y_ic2 = Y_ic + gap_y_ic[FOV_num]
XY_ic_poly = Polygon(np.column_stack((X_ic, Y_ic)))
XY_ic_poly2 = Polygon(np.column_stack((X_ic2, Y_ic2)))
overlapRegion_fc, modifiedPolygon_fc = overlap_FOV(XY_ic_poly, XY_ic_poly2)

X_ic = []
Y_ic = []
polygon_to_xy(modifiedPolygon_fc, X_ic, Y_ic)
area_ic = modifiedPolygon_fc.area
energy_after_ic = area_ic / 12.5579361  # normalize by input pupil area

# === Find TIR start index ===
for i in range(1001):
    test_X = X_ic + i * gap_x_ic[FOV_num]
    test_Y = Y_ic + i * gap_y_ic[FOV_num]
    test_poly = Polygon(np.column_stack((test_X.T, test_Y.T)))
    test_poly2 = Polygon(np.column_stack((points_x_FOV_fc_all, points_y_FOV_fc_all)))
    overlapRegion_fc, _ = overlap_FOV(test_poly2, test_poly)

    if not overlapRegion_fc.is_empty:
        num_st_TIR = i
        break
    else:
        num_st_TIR = None  # in case no overlap found

print(f"Energy after input coupler: {energy_after_ic:.4f}")
print(f"Start TIR step index: {num_st_TIR}")

X_st = X_ic[0]+gap_x_ic[FOV_num]*2
Y_st = Y_ic[0]+gap_y_ic[FOV_num]*2
# plt.fill(X_st, Y_st, 'orange', alpha=0.9)
# Region_test = Polygon(np.column_stack((X_st.T, Y_st.T)))
# FOV_fc_region_test = Polygon(np.column_stack((points_x_FOV_fc[0], points_y_FOV_fc[0])))
# overlapRegion_oc2, modifiedPolygon_oc2 = overlap_FOV(FOV_fc_region_test, Region_test)
# xs_list = []
# ys_list = []
# polygon_to_xy(modifiedPolygon_oc2, xs_list, ys_list)
# plt.fill(xs_list[0], ys_list[0], 'orange', alpha=0.9)
# xs_list = []
# ys_list = []
# polygon_to_xy(overlapRegion_oc2, xs_list, ys_list)
# plt.fill(xs_list[0]+gap_x_fc[FOV_num], ys_list[0]+gap_y_fc[FOV_num], 'orange', alpha=0.5)
# plt.fill(xs_list[0]+2*gap_x_ic[FOV_num], ys_list[0]+2*gap_y_ic[FOV_num], 'orange', alpha=0.5)
# plt.fill(xs_list[0]+gap_x_fc[FOV_num]+gap_x_ic[FOV_num], ys_list[0]+gap_y_fc[FOV_num]+gap_y_ic[FOV_num], 'orange', alpha=0.5)

# Lens size
width = 58   # mm
height = 42  # mm
a = width / 2
b = height / 2
n = 4  # Superellipse exponent

# Superellipse full shape
theta_main = np.linspace(0, 2 * np.pi, 500)
x_main = a * np.sign(np.cos(theta_main)) * np.abs(np.cos(theta_main))**(2/n)
y_main = b * np.sign(np.sin(theta_main)) * np.abs(np.sin(theta_main))**(2/n)

# Half-circle notch on left side
r = b / 2
theta_half = np.linspace(np.pi/2, 3*np.pi/2, 100)
x_half = -a + r * np.cos(theta_half)+6
y_half = r * np.sin(theta_half)+1.5

# Combine all candidate boundary points
x_all = np.concatenate([x_main, x_half])
y_all = np.concatenate([y_main, y_half])
points = np.vstack((x_all, y_all)).T

# Compute convex hull to extract outer edge
hull = ConvexHull(points)
hull_points = points[hull.vertices]

plt.fill(hull_points[:, 0], hull_points[:, 1]+13, color='lightblue', alpha=0.3)


# show the plot
plt.xlim(-35, 30)
plt.ylim(-10, 35)
plt.title("Waveguide Design Plot", fontsize=14, weight='bold')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.tick_params(labelsize=10)
plt.show()