import bpy
import numpy as np
import mathutils
from scipy.spatial import SphericalVoronoi, geometric_slerp
from shapely.geometry import Polygon


def adjust_length(points, radii):
    adjusted_points = []
    for point in points:
        point = point * radii / np.linalg.norm(point)
        adjusted_points.append(point)
    return adjusted_points


def generate_random_points(constraint, num_points, radii):  # generate points given a 3 plane constraint
    cross_product = np.cross(constraint[1], constraint[2])
    if cross_product[2] < 0:
        cross_product = cross_product * (-1)
    points = []
    while len(points) < num_points:
        phi = np.random.uniform(-1 * np.pi / 5, 0)
        costheta = np.random.uniform(-1, 0)
        theta = np.arccos(costheta)
        x = np.sin(theta) * np.cos(phi) * radii
        y = np.sin(theta) * np.sin(phi) * radii
        z = np.cos(theta) * radii
        point = np.array([x, y, z])
        if np.inner(point, cross_product) < 0:
            points.append(point)
    return np.array(points)


def remove_outside_points(points, focus_point):
    cross_product_1 = np.cross(focus_point[0], focus_point[1])
    if cross_product_1[1] < 0:
        cross_product_1 = cross_product_1 * (-1)
    cross_product_2 = np.cross(focus_point[1], focus_point[2])
    if cross_product_2[2] < 0:
        cross_product_2 = cross_product_2 * (-1)
    i = 0
    while i < len(points):
        if (np.inner(cross_product_1, points[i]) <= 0) or (np.inner(cross_product_2, points[i]) >= 0) or (points[i][1] >= 0):
            del points[i]
        else:
            i = i + 1
    return points


def rodrigues_rotation(axis, point, angle):
    axis = axis / np.linalg.norm(axis)
    rotated = np.cos(angle) * point + np.sin(angle) * np.cross(axis, point) + (1 - np.cos(angle)) * np.inner(axis, point) * axis
    return rotated


def rotation(points, vertices, focus_point):
    # Rotate points such that they fill all the sphere
    full_face_points = []
    for i in range(3):
        for point in points:
            full_face_points.append(rodrigues_rotation(focus_point[2], point, 2 * i * np.pi / 3))
    full_points = []
    for i in range(5):
        for point in full_face_points:
            face_point1 = rodrigues_rotation(vertices[0], point, 2 * np.pi * i / 5)
            full_points.append(face_point1)
            full_points.append(rodrigues_rotation(vertices[i + 1], face_point1, 2 * np.pi / 5))
            face_point2 = rodrigues_rotation(vertices[i + 1], face_point1, 4 * np.pi / 5)
            full_points.append(face_point2)
            full_points.append(rodrigues_rotation(vertices[i + 6], face_point2, 2 * np.pi / 5))
    return full_points


def lloyd_relaxation_algorithm(iteration, points, radius, center, vertices, focus_point):
    i = 1
    point1 = np.array([])
    while i < iteration:
        point1 = points.copy()
        sv = SphericalVoronoi(points, radius, center)
        sv.sort_vertices_of_regions()
        centroids = []
        for region in sv.regions:
            cross_product = np.cross(sv.vertices[region[0]] - sv.vertices[region[1]], sv.vertices[region[0]] - sv.vertices[region[2]])
            flat_region_point = [sv.vertices[region[0]], sv.vertices[region[1]], sv.vertices[region[2]]]
            for remaining_point in region[3:]:
                if np.linalg.norm(cross_product) == 0:
                    project_point = sv.vertices[remaining_point]
                else:
                    project_point = sv.vertices[remaining_point] - cross_product * np.inner(cross_product, sv.vertices[remaining_point] - sv.vertices[region[1]]) / np.linalg.norm(cross_product) ** 2
                flat_region_point.append(project_point)
            flat_region_point_xy = [arr[:2] for arr in flat_region_point]
            flat_region_point_yz = [arr[1:] for arr in flat_region_point]
            polygon_xy = Polygon(flat_region_point_xy)
            centroid_xy = polygon_xy.centroid
            polygon_yz = Polygon(flat_region_point_yz)
            centroid_yz = polygon_yz.centroid
            centroid_point = np.array([centroid_xy.x, centroid_xy.y, centroid_yz.y])
            projected_central_point = centroid_point * radius / np.linalg.norm(centroid_point)
            centroids.append(projected_central_point)
        points = list(np.array(centroids.copy()))
        points = remove_outside_points(points, focus_point)  # input list, output list
        points = rotation(points, vertices, focus_point)  # input list, output list
        points = np.array(adjust_length(points, radius))
        print(i)
        i = i + 1
    return point1


def draw_radius(points, factor):
    points = list(points)
    radius = []
    for i in range(len(points)):
        distance = []
        points1 = points.copy()
        del points1[i]
        for j in range(len(points1)):
            distance.append(np.linalg.norm(points[i] - points1[j]))
        radius.append(min(distance) / 2)
    return np.array(radius) * factor


def cut_hole(sphere, obj):
    bool_modifier = sphere.modifiers.new(name="Boolean", type='BOOLEAN')
    bool_modifier.object = obj
    bool_modifier.operation = 'DIFFERENCE'
    bpy.context.view_layer.objects.active = sphere
    bpy.ops.object.modifier_apply(modifier=bool_modifier.name)
    bpy.data.objects.remove(obj)


def create_hole_cut(shape, sphere_radius, sphere, radius, location, height, angle):
    if shape == "sphere":
        small_sphere_radius = height - sphere_radius + np.divide(2 * height * sphere_radius - np.square(height), (2 * (height - sphere_radius + np.sqrt(abs(np.square(sphere_radius) - np.square(radius))))))
        obj = create_smooth_sphere(true_radius=small_sphere_radius, location=location * (1 + (small_sphere_radius - height) / sphere_radius))
    elif shape == "cone":
        bottom_radius = (height * radius) / (height - sphere_radius + np.sqrt(np.square(sphere_radius)-np.square(radius)))
        bpy.ops.mesh.primitive_cone_add(vertices=200, radius1=2*bottom_radius, radius2=0, depth=2*height, location = location) #radius will be slightly smaller than expected
        obj = bpy.context.object
        rotate(obj, location)
    elif shape == "frustum":
        bottom_radius = radius + (sphere_radius - np.sqrt(np.square(sphere_radius)-np.square(radius))) / np.tan(angle)
        bpy.ops.mesh.primitive_cone_add(vertices=200, radius1=bottom_radius + height / np.tan(angle), radius2=bottom_radius - height / np.tan(angle), depth=2*height, location = location) #radius will be slightly smaller than expected
        obj = bpy.context.object
        rotate(obj, location)
    else:
        return "Invalid shape"
    cut_hole(sphere, obj)


def rotate(obj, location):
    current_direction = obj.matrix_world.to_quaternion() @ mathutils.Vector((0, 0, 1))
    rotation_quaternion = current_direction.rotation_difference(-1 * location)
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = rotation_quaternion


def create_smooth_sphere(true_radius, location):
    bpy.ops.mesh.primitive_ico_sphere_add(radius=true_radius, subdivisions=4, location=location)
    cube = bpy.context.object
    subsurf_modifier = cube.modifiers.new(name="Subdivision", type='SUBSURF')
    subsurf_modifier.levels = 1
    bpy.ops.object.modifier_apply(modifier=subsurf_modifier.name)
    vertices = []
    distance = []
    for v in cube.data.vertices:
        v = np.array(v.co)
        vertices.append(v)
        distance.append(np.linalg.norm(v))
    one_radius = sum(distance) / len(distance)
    cube.scale = true_radius * (cube.scale / one_radius)
    return cube



sphere_radius = 42.6
num_points = 6
iteration = 2000
factor = 0.9
depth = 0.5
add_vertices = True
shape = "cone"
angle = 70 

bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

bpy.ops.mesh.primitive_ico_sphere_add(radius=sphere_radius, subdivisions=1, location=(0, 0, 0))
obj = bpy.context.active_object
vertices = [np.array(v.co) for v in obj.data.vertices]
vertices = adjust_length(vertices, sphere_radius)

bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

# Draw the center points of each faces
center_points = []
for i in range(1, 11):
    if 1 <= i <= 4:
        center_points.append((vertices[0] + vertices[i] + vertices[i + 1]) / 3)
        center_points.append((vertices[i + 5] + vertices[i] + vertices[i + 1]) / 3)
    elif i == 5:
        center_points.append((vertices[0] + vertices[i] + vertices[1]) / 3)
        center_points.append((vertices[10] + vertices[i] + vertices[1]) / 3)
    elif 6 <= i <= 9:
        center_points.append((vertices[11] + vertices[i] + vertices[i + 1]) / 3)
        center_points.append((vertices[i - 4] + vertices[i] + vertices[i + 1]) / 3)
    else:
        center_points.append((vertices[11] + vertices[i] + vertices[6]) / 3)
        center_points.append((vertices[1] + vertices[i] + vertices[6]) / 3)
center_points = adjust_length(center_points, sphere_radius)

# Select the appropriate triangle to generate random points
focus_point = [vertices[0], vertices[1], center_points[8]]
points = list(generate_random_points(focus_point, num_points, sphere_radius))
points = adjust_length(points, sphere_radius)
# Rotate points such that they fill all the sphere
full_points = rotation(points, vertices, focus_point)
full_points = adjust_length(full_points, sphere_radius)

result = lloyd_relaxation_algorithm(iteration, np.array(full_points), sphere_radius, np.array([0, 0, 0]), vertices, focus_point)
radii = draw_radius(result, factor)

if add_vertices == True:
    final_distance = []
    result_list = list(result)
    radii_list = list(radii)
    for vertice in vertices:
        distance = []
        for i in range(len(result_list)):
            distance.append(np.linalg.norm(vertice-result_list[i])-radii_list[i])
        final_distance.append(min(distance)*factor*0.8)
    result = np.array(result_list + vertices)
    radii = np.array(radii_list + final_distance)

sphere = create_smooth_sphere(true_radius=sphere_radius, location=(0, 0, 0))
for i in range(len(list(result))):
    print(i)
    create_hole_cut(shape, sphere_radius, sphere, radii[i], result[i], depth, angle)
