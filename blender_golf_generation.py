bl_info = {
    "name": "Golf",
    "author": "Tony",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "View3D > Sidebar > Golf",
    "description": "Adjust golf",
    "category": "3D View",
}

import os
import bpy
import bmesh
import numpy as np
import mathutils
from mathutils import Quaternion, Vector
from scipy.spatial import SphericalVoronoi
from shapely.geometry import Polygon
from bpy.props import FloatProperty


root_3 = np.sqrt(3)
one_over_root_3 = np.sqrt(3) / 3
root_2 = np.sqrt(2)
one_over_root_2 = np.sqrt(2) * 0.5
#[cos(θ), sin(θ), tan(θ)]
degree_120 = [-0.5, 0.5 * root_3, -1 * root_3]
degree_240 = [-0.5, -0.5 * root_3, root_3]


def adjust_length(points, radii):
    # adjust every point such that they lie on the sphere
    adjusted_points = []
    for point in points:
        point = point * radii / np.linalg.norm(point)
        adjusted_points.append(point)
    return adjusted_points


def adjust_vertices(geometry, scale):
    for vertice in geometry.data.vertices:
        v = np.array(vertice.co)
        vertice.co = Vector(v * scale/ np.linalg.norm(v))
    return geometry


def generate_random_points(num_points): 
    # generate points on a designated area
    points = []
    while len(points) < num_points:
        phi = np.random.uniform(-1 * np.pi / 4, np.pi / 4)
        costheta = np.random.uniform(0, 1)
        theta = np.arccos(costheta)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
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
        half_points = [point, rodrigues_rotation(np.array([1, 1, 1]) * one_over_root_3, point, degree_120), rodrigues_rotation(np.array([1, 1, 1]) * one_over_root_3, point, degree_240)]
        full_points.extend(half_points)
        full_points.extend([np.array([1, -1, -1]) * half_points[0], np.array([-1, 1, -1]) * half_points[1], np.array([-1, -1, 1]) * half_points[2]])
    return full_points


def lloyd_relaxation_algorithm(iteration, points):
    i = 1
    point1 = np.array([])
    while i < iteration:
        point1 = points.copy()
        sv = SphericalVoronoi(points, 1, np.array([0, 0, 0]))  # the OP function, without it the project would fail, huge thanks to scipy!
        sv.sort_vertices_of_regions()  
        centroids = []
        for region in sv.regions:  # after voronoi cells are generated, the next step is to generate centroids for each cell, this for loop does exactly that
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
            centroids.append(centroid_point)
        points = list(np.array(centroids.copy()))
        points = remove_outside_points(points)
        points = rotation(points)
        points = adjust_length(points, 1)
        print("performing lloyd relaxation: " + str(i) + "/" + str(iteration))
        i = i + 1
    return point1


def create_smooth_sphere():
    bpy.ops.mesh.primitive_ico_sphere_add(radius=1, subdivisions=4, location=(0, 0, 0)) # adjust if needed, the higher, the smoother
    cube = bpy.context.object
    subsurf_modifier = cube.modifiers.new(name="Subdivision", type='SUBSURF')
    subsurf_modifier.levels = 4 # adjust if needed, the higher, the smoother
    bpy.ops.object.modifier_apply(modifier=subsurf_modifier.name)
    cube = adjust_vertices(cube, 1)
    return cube


def prepare(center_points, radii):
    center_points = np.array(center_points)
    vertices = sphere.data.vertices
    for vertice in vertices:
        v = np.array(vertice.co)
        dist = v - center_points
        dist_norm = np.linalg.norm(dist, axis=1)
        closest_point_pos = np.argmin(dist_norm)
        closest_center_point = center_points[closest_point_pos]
        dot = np.clip(np.dot(closest_center_point, v), -1, 1)
        dev_angle = np.arccos(dot)
        perp_dist = np.linalg.norm(v - closest_center_point * dot)
        factor = perp_dist / radii[closest_point_pos]
        prep_list.append([closest_point_pos, factor, np.tan(dev_angle), np.sin(dev_angle)]) #maximum radius on true sphere, factor, 
        #After filling in the vertice point, the next step is to dig holes in it, which requires slider


def draw_max_radius(points):
    # draw radius given the position of each dimple center
    radius = []
    for i in range(len(points)):
        distance = []
        points1 = points.copy()
        del points1[i]
        for j in range(len(points1)):
            distance.append(np.linalg.norm(points[i] - points1[j]))
        radius.append(min(distance) * 0.5)
    return radius


def function_1(context):
    global prep_list, sphere, max_rad, min_rad, radii, dimples_per_unit
    
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    prep_list = []
    dimples_per_unit = 20
    iteration = 500

    points = list(generate_random_points(dimples_per_unit))
    full_points = rotation(points)
    result = lloyd_relaxation_algorithm(iteration, np.array(full_points))
    radii = draw_max_radius(result)

    max_rad = max(radii)
    min_rad = min(radii)

    sphere = create_smooth_sphere()
    
    prepare(result, radii)
    print("Done!")


def function_2(factor, depth):
    if not sphere:
        print("sphere not exist")
        return
    
    hole_radius_max = factor * max_rad
    hole_radius_min = factor * min_rad
    threshold_min = 1 - np.sqrt(1 - np.square(hole_radius_max))
    threshold_max = 0.5 * hole_radius_min
    
    if (threshold_max < depth) or (depth < threshold_min):
        print(f"either the hole was too flat or too curved, depth should be between {threshold_min:.4f} and {threshold_max:.4f}")
        return
    
    hole_radius_list = np.array([r * factor for r in radii])
    cut_sphere_radius_list = list(depth - 1 + np.divide(2 * depth - np.square(depth), (2 * (depth - 1 + np.sqrt(1 - np.square(hole_radius_list))))))
    shrink_factor_list = []
    
    for l in prep_list:
        if l[1] >= factor:
            shrink_factor_list.append(1)
            continue
        cut_sphere_radius = cut_sphere_radius_list[l[0]]
        sin_theta_1 = (1 + (1 - depth) / cut_sphere_radius) * l[3]
        cos_theta_1 = -1 * np.cos(np.arcsin(sin_theta_1)) #warning: this might not be uniformly -1 or 1, one must check to see if the angle is <90 or >90.
        shrink_factor = (cos_theta_1 + sin_theta_1 / l[2]) * cut_sphere_radius
        shrink_factor_list.append(shrink_factor)
    
    for ver, sf in zip(sphere.data.vertices, shrink_factor_list):
        ver.co = sf * ver.co / ver.co.length
        

def function_3():
    sphere_radius = 21.3
    if not sphere:
        print("sphere not exist")
        return
    for vertice in sphere.data.vertices:
        v = np.array(vertice.co)
        vertice.co = Vector(v * sphere_radius)
    factor = bpy.context.scene.factor
    depth = bpy.context.scene.depth
    true_depth = depth * sphere_radius
    total_num = 24 * dimples_per_unit
    path = f"~/Downloads/sphere_radius={sphere_radius:.3f},factor={factor:.3f},depth={true_depth:.3f},num={total_num}.stl"
    filepath = os.path.expanduser(path)

    sphere.select_set(True)
    bpy.context.view_layer.objects.active = sphere

    bpy.ops.export_mesh.stl(
        filepath=filepath,
        use_selection=True,
        ascii=False,
        use_mesh_modifiers=True
    )
    for vertice in sphere.data.vertices:
        v = np.array(vertice.co)
        vertice.co = Vector(v / sphere_radius)
    print(f"Exported '{sphere.name}' to {filepath}")


def slider_update(self, context):
    factor = context.scene.factor
    depth = context.scene.depth
    function_2(factor, depth)


class WM_OT_generate_distribution(bpy.types.Operator):
    bl_idname = "wm.generate_distribution"
    bl_label = "Generate distribution"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        result = function_1(context)
        self.report({'INFO'}, f"result: {result}")
        return {'FINISHED'}
    

class WM_OT_export(bpy.types.Operator):
    bl_idname = "wm.export_stl"
    bl_label = "Export to STL"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        result = function_3()
        self.report({'INFO'}, f"result: {result}")
        return {'FINISHED'}


class VIEW3D_PT_golf(bpy.types.Panel):
    bl_label = "Golf"
    bl_idname = "VIEW3D_PT_golf_generation"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Golf"  # the tab name in the sidebar

    def draw(self, context):
        layout = self.layout
        layout.operator(WM_OT_generate_distribution.bl_idname, text="Generate distribution")
        layout.prop(context.scene, "factor", slider=True)
        layout.prop(context.scene, "depth", slider=True)
        layout.operator(WM_OT_export.bl_idname, text="Export to STL")


classes = (
    WM_OT_generate_distribution,
    WM_OT_export, 
    VIEW3D_PT_golf,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
        
    bpy.types.Scene.factor = FloatProperty(
        name="Factor",
        description="Adjust factor",
        default=0.8,
        min=0.6,
        max=1.0,
        step=0.001,
        precision=3,
        update=slider_update,  # this function will be called on change
    )
    bpy.types.Scene.depth = FloatProperty(
        name="Depth",
        description="Adjust depth",
        default=0.5,
        min=0.0,
        max=0.05,
        step=0.001,
        precision=3,
        update=slider_update,  # this function will be called on change
    )


def unregister():
    # Collect candidates first to avoid mutating the subclasses() list while iterating.
    for pt in bpy.types.Panel.__subclasses__():
        if (pt.bl_space_type == 'VIEW_3D') and (pt.bl_label == "Golf"):
            if "bl_rna" in pt.__dict__: 
                bpy.utils.unregister_class(pt)


if __name__ == "__main__":
    #Assume the sphere is 1, it only changes to the current sphere_radius when you try to export it, please write a code to export
    register()
