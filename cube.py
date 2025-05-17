"""
generate golfball using cube platonic solid as framework
copy-paste the code into blender's text editor to run it, make sure the viewport is in "object mode"
"""
import bpy
import numpy as np
import mathutils
from mathutils import Quaternion, Vector
from scipy.spatial import SphericalVoronoi
from shapely.geometry import Polygon


def adjust_length(points, radii):
    # adjust every point such that they lie on the sphere
    adjusted_points = []
    for point in points:
        point = point * radii / np.linalg.norm(point)
        adjusted_points.append(point)
    return adjusted_points


def generate_random_points(num_points, radii): 
    # generate points on a designated area
    points = []
    while len(points) < num_points:
        phi = np.random.uniform(-1 * np.pi / 4, np.pi / 4)
        costheta = np.random.uniform(0, 1)
        theta = np.arccos(costheta)
        x = np.sin(theta) * np.cos(phi) * radii
        y = np.sin(theta) * np.sin(phi) * radii
        z = np.cos(theta) * radii
        point = np.array([x, y, z])
        if (np.inner(point, np.array([1, 1, 0])) > 0) and (np.inner(point, np.array([-1, 0, 1])) > 0) and (np.inner(point, np.array([1, -1, 0])) > 0):
            points.append(point)
    return np.array(points)


def remove_outside_points(points):
    # remove all points outside designated area
    i = 0
    while i < len(points):
        if (np.inner(np.array([1, 1, 0]), points[i]) <= 0) or (np.inner(np.array([-1, 0, 1]), points[i]) <= 0) or (np.inner(np.array([1, -1, 0]), points[i]) <= 0):
            del points[i]
        else:
            i = i + 1
    return points


def rodrigues_rotation(axis, point, angle):
    rotated = angle[0] * point + angle[1] * np.cross(axis, point) + (1 - angle[0]) * np.inner(axis, point) * axis
    return rotated


def rotation(points):
    # copy and rotate these points such that they fill the whole sphere
    full_face_points = []
    angle_point = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    for point in points:
        for angle in angle_point:
            full_face_points.append(rodrigues_rotation(np.array([0, 0, 1]), point, angle))
        
    full_points = []
    for point in full_face_points:
        full_points.append(point)
        full_points.append(point * np.array([-1, 1, -1]))
        full_points.append(rodrigues_rotation(np.array([1, 1, 1]) * one_over_root_3, point, degree_120))
        full_points.append(rodrigues_rotation(np.array([1, 1, 1]) * one_over_root_3, point, degree_240))
        full_points.append(rodrigues_rotation(np.array([-1, -1, 1]) * one_over_root_3, point, degree_120))
        full_points.append(rodrigues_rotation(np.array([-1, -1, 1]) * one_over_root_3, point, degree_240))
    return full_points


def lloyd_relaxation_algorithm(iteration, points, radius, center):
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
        points = remove_outside_points(points)
        points = adjust_length(points, radius)
        points = np.array(rotation(points))
        print("performing lloyd relaxation: " + str(i) + "/" + str(iteration))
        i = i + 1
    return point1


def draw_radius(points, factor):
    # draw radius given the position of each dimple center
    points = list(points)
    radius = []
    for i in range(len(points)):
        distance = []
        points1 = points.copy()
        del points1[i]
        for j in range(len(points1)):
            distance.append(np.linalg.norm(points[i] - points1[j]))
        radius.append(min(distance) * 0.5)
    return np.array(radius) * factor


def cut(sphere, obj):
    # cut holes on the sphere to generate dimples
    bool_modifier = sphere.modifiers.new(name="Boolean", type='BOOLEAN')
    bool_modifier.object = obj
    bool_modifier.operation = 'DIFFERENCE'
    bpy.context.view_layer.objects.active = sphere
    bpy.ops.object.modifier_apply(modifier=bool_modifier.name)
    bpy.data.objects.remove(obj)


def create_hole_cut(shape, sphere_radius, sphere, radius, location, height, angle):
    # generate a dimple on the sphere
    if shape == "sphere":
        small_sphere_radius = height - sphere_radius + np.divide(2 * height * sphere_radius - np.square(height), (2 * (height - sphere_radius + np.sqrt(np.square(sphere_radius) - np.square(radius)))))
        obj = create_smooth_sphere(true_radius=small_sphere_radius, location=location * (1 + (small_sphere_radius - height) / sphere_radius))
    elif shape == "cone":
        bottom_radius = (height * radius) / (height - sphere_radius + np.sqrt(np.square(sphere_radius)-np.square(radius)))
        bpy.ops.mesh.primitive_cone_add(vertices=500, radius1=2*bottom_radius, radius2=0, depth=2*height, location = location) # adjust if needed
        obj = bpy.context.object
        rotate(obj, location)
    elif shape == "frustum":
        bottom_radius = radius + (sphere_radius - np.sqrt(np.square(sphere_radius)-np.square(radius))) / degree_frustum_angle[2]
        bpy.ops.mesh.primitive_cone_add(vertices=500, radius1=bottom_radius + height / degree_frustum_angle[2], radius2=bottom_radius - height / degree_frustum_angle[2], depth=2*height, location = location) # adjust if needed
        obj = bpy.context.object
        rotate(obj, location)
    else:
        raise ValueError("Invalid shape! Please choose between sphere, cone, and frustum")
    cut(sphere, obj)


def rotate(obj, location):
    # for cone and frustum, its orientation needs to be adjusted to point towards the center
    current_direction = obj.matrix_world.to_quaternion() @ mathutils.Vector((0, 0, 1))
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = current_direction.rotation_difference(-1 * location)


def create_smooth_sphere(true_radius, location):
    bpy.ops.mesh.primitive_ico_sphere_add(radius=true_radius, subdivisions=4, location=location) # adjust if needed
    cube = bpy.context.object
    subsurf_modifier = cube.modifiers.new(name="Subdivision", type='SUBSURF')
    subsurf_modifier.levels = 3 # adjust if needed
    bpy.ops.object.modifier_apply(modifier=subsurf_modifier.name)
    vertices = []
    distance = []
    for vertice in cube.data.vertices:
        v = np.array(vertice.co)
        vertice.co = Vector(true_radius * v / np.linalg.norm(v))
    return cube


def remove(sphere, sphere_radius):
    # trim the sphere down to the designated part
    locations = [(root_2*sphere_radius*0.75, 0, -1*root_2*sphere_radius*0.75), (-1*root_2*sphere_radius*0.75, -1*root_2*sphere_radius*0.75, 0), (-1*root_2*sphere_radius*0.75, root_2*sphere_radius*0.75, 0)]
    axis = [np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([0, 0, 1])]
    for i in range(3):
        bpy.ops.mesh.primitive_cube_add(size=3*sphere_radius, location=locations[i])
        cutter= bpy.context.object
        for vertice in cutter.data.vertices:
            v = np.array(vertice.co)
            vertice.co = Vector(rodrigues_rotation(axis[i], v, [one_over_root_2, one_over_root_2]))
        cut(sphere, cutter)
        

def piece_together(sphere):
    # copy, rotate, and piece together these designated part to form a complete golfball
    obj_collection = [sphere]
    angle_list_1 = [[0,1], [-1,0], [0,-1]]

    for i in range(3):
        print("copying small pieces: " + str(i) + "/" + str(2))
        copy = sphere.copy()
        copy.data = sphere.data.copy()
        bpy.context.collection.objects.link(copy)
        obj_collection.append(copy)
        
        for v in copy.data.vertices:
            v.co = Vector(rodrigues_rotation(np.array([0, 0, 1]), np.array(v.co), angle_list_1[i]))
        copy.data.update()

    bpy.ops.object.select_all(action='DESELECT')
    for obj in obj_collection:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = obj_collection[0]
    bpy.ops.object.join()
    
    joined_object = bpy.context.active_object
    obj_collection[:] = [joined_object]
    for mesh in bpy.data.meshes:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    
    points_list_raw = [np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([-1, -1, 1]), np.array([-1, -1, 1]), np.array([1, 0, 0])]
    points_list = [point * one_over_root_3 for point in points_list_raw[:-1]] + [points_list_raw[-1]]
    angle_list_2 = [degree_120, degree_240, degree_120, degree_240, [-1,0]]
    
    for i in range(5):
        print("copying big pieces: " + str(i) + "/" + str(4))
        joined_copy = joined_object.copy()
        joined_copy.data = joined_object.data.copy()
        bpy.context.collection.objects.link(joined_copy)
        obj_collection.append(joined_copy)

        for v in joined_copy.data.vertices:
            v.co = Vector(rodrigues_rotation(points_list[i], np.array(v.co), angle_list_2[i]))
        joined_copy.data.update()

    bpy.ops.object.select_all(action='DESELECT')
    for obj in obj_collection:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = obj_collection[0]
    bpy.ops.object.join()
    
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=0.0001)  # adjust if needed
    bpy.ops.object.mode_set(mode='OBJECT')
    
    
def remove_outside_points_2(points, radii):
    i = 0
    while i < len(points):
        if (-0.1 * sphere_radius <= points[i][0] <= 0.8 * sphere_radius) and (-0.65 * sphere_radius <= points[i][1] <= 0.65 * sphere_radius) and (points[i][2] >= 0.55 * sphere_radius):
            i = i + 1
        else:
            del points[i]
            del radii[i]
    return np.array(points), np.array(radii)



'''
Explanation for each variable:
sphere_radius: the radius of the golfball
dimples_per_unit: this does NOT refer to the total number of dimple generated on the golfball
* for this file, the total dimple on the sphere should be: dimples_per_unit * 24 (see readme file for more info)
iteration: the more iteration the algorithm performs, the more evenly distributed the dimples will be (see readme file for more info)
radius_factor: given the random nature of the distribution algorithm, there won't be a unified radius for every dimple, rather, a factor is needed to deduce the radius
* each dimple radius will be calculated wrt the distance to the nearest dimple center, and the factor should be restricted between 0 and 1 (see readme file for more info)
depth: the unified depth of every dimple
shape: the shape of the dimple, there are three shapes to choose from: sphere, cone, and frustum
angle: this variable only applies to frustum, the bigger the angle, the steeper the frustum, the steeper the dimple edge. Angle should be between 0 and 90 degrees
'''

#input
sphere_radius = 42.6
dimples_per_unit = 20
iteration = 1000
radius_factor = 0.95
depth = 0.5
shape = "cone" 
angle = 70 

#constant
root_3 = np.sqrt(3)
one_over_root_3 = np.sqrt(3) / 3
root_2 = np.sqrt(2)
one_over_root_2 = np.sqrt(2) * 0.5
#[cos(θ), sin(θ), tan(θ)]
degree_frustum_angle = [np.cos(np.radians(angle)), np.sin(np.radians(angle)), np.tan(np.radians(angle))]
degree_120 = [-0.5, 0.5 * root_3, -1 * root_3]
degree_240 = [-0.5, -0.5 * root_3, root_3]

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# first, random points will be generated on the designated part of the region
print("generating random points")
points = list(generate_random_points(dimples_per_unit, sphere_radius))
points = adjust_length(points, sphere_radius)

# then, copy and rotate these points such that they fill the whole sphere
full_points = rotation(points)
full_points = adjust_length(full_points, sphere_radius)

# performing the lloyd relaxation algorithm
result = lloyd_relaxation_algorithm(iteration, np.array(full_points), sphere_radius, np.array([0, 0, 0]))
radii = draw_radius(result, radius_factor)

# generate golfball sphere, then trim it down to only the designated part
sphere = create_smooth_sphere(true_radius=sphere_radius, location=(0, 0, 0))
result, radii = remove_outside_points_2(list(result), list(radii))
remove(sphere, sphere_radius)

# generate dimples in the designated part
for i in range(len(list(result))):
    print("generating dimples: " + str(i) + "/" + str(len(list(result))))
    create_hole_cut(shape, sphere_radius, sphere, radii[i], result[i], depth, degree_frustum_angle)

# copy, rotate, and piece together these designated part to form a complete golfball
piece_together(sphere)
