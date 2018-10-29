from utils import parse_utils
import torch
from ray_tracing import ray_surface_intersection
from control import device_control
from . import shape


class Sphere(shape.Shape):
    @staticmethod
    def parse_from_description(description_list):
        sphere = Sphere()
        for description in description_list:
            if description[0] == "center":
                sphere.center = parse_utils.parse_tensor_from_string_toks(description[1: ])
            elif description[0] == "radius":
                sphere.radius = parse_utils.parse_tensor_from_string_toks(description[1:])
            else:
                sphere.load_default_property(description)
        return sphere
    
    def __init__(self):
        super(Sphere, self).__init__()


    def get_color_at_point(self, p):
        return self.color

    def find_intersections(self, rays):
        a = rays.directions.dot(rays.directions)
        o_minus_c = rays.starts - self.center
        b = 2 * rays.directions.dot(o_minus_c)
        c = o_minus_c.dot(o_minus_c) - self.radius * self.radius
        no_intersection_mask = b * b < 4.0 * a * c
        root_del = torch.sqrt(b * b - 4 * a * c)
        t1 = (-b + root_del) / (2.0 * a)
        t2 = (-b - root_del) / (2.0 * a)
        bad_t_mask = (t1 < 0) & (t2 < 0)
        no_intersection_mask = no_intersection_mask | bad_t_mask
        intersection = ray_surface_intersection.RaySurfaceIntersection()
        intersection.intersected = 1 - no_intersection_mask
        inf_tensor = device_control.get_device_float32_array([t1.shape[0]], 1e30)
        intersection.t = torch.min(torch.where(t1 < 0, inf_tensor, t1), torch.where(t2 < 0, inf_tensor, t2))
        intersection.intersection_points = rays.get_point_on_line(intersection.t)
        intersection.intersection_normals = (intersection.intersection_points - self.center).unit_vectors()
        intersection.reflection_directions = rays.directions.get_reflection_directions(intersection.intersection_normals)
        intersection.incident_rays = rays
        intersection.colors_at_intersection = self.get_color_at_point(intersection.intersection_points)
        return intersection

    def get_intersection_t(self, rays):
        a = rays.directions.dot(rays.directions)
        o_minus_c = rays.starts - self.center
        b = 2 * rays.directions.dot(o_minus_c)
        c = o_minus_c.dot(o_minus_c) - self.radius * self.radius
        no_intersection_mask = b * b < 4.0 * a * c
        root_del = torch.sqrt(b * b - 4 * a * c)
        t1 = (-b + root_del) / (2.0 * a)
        t2 = (-b - root_del) / (2.0 * a)
        bad_t_mask = (t1 < 0) & (t2 < 0)
        no_intersection_mask = no_intersection_mask | bad_t_mask
        intersected = 1 - no_intersection_mask
        inf_tensor = device_control.get_device_float32_array([t1.shape[0]], 1e30)
        t = torch.min(torch.where(t1 < 0, inf_tensor, t1), torch.where(t2 < 0, inf_tensor, t2))
        t = torch.where(intersected, t, device_control.get_device_float32_array([t.shape[0]], 1e-10))
        return t