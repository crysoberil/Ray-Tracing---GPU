from . import shape
from utils import parse_utils
from control import device_control
# from ray_tracing import ray_surface_intersection
from geometric_primitives import plane, vector
import torch


class Triangle(shape.Shape):
    @staticmethod
    def parse_from_description(description_list):
        triangle = Triangle()
        for description in description_list:
            if description[0] == "a":
                triangle.a = vector.Vector3(device_control.to_gpu_if_possible(parse_utils.parse_tensor_from_string_toks(description[1: ])).view(1, -1))
            elif description[0] == "b":
                triangle.b = vector.Vector3(device_control.to_gpu_if_possible(parse_utils.parse_tensor_from_string_toks(description[1: ])).view(1, -1))
            elif description[0] == "c":
                triangle.c = vector.Vector3(device_control.to_gpu_if_possible(parse_utils.parse_tensor_from_string_toks(description[1: ])).view(1, -1))
            else:
                triangle.load_default_property(description)
        triangle.plane = plane.Plane(triangle.a, triangle.b, triangle.c)
        return triangle

    def __init__(self):
        super(Triangle, self).__init__()

    def get_color_at_point(self, p):
        return self.color

    def find_intersections(self, rays):
        intersection = self.plane.find_intersections(rays)
        p = intersection.intersection_points
        c0 = p - self.a
        c1 = p - self.b
        c2 = p - self.c
        edge0, edge1, edge2 = self.b - self.a, self.c - self.b, self.a - self.c
        inside0 = self.plane.normal_directions.dot(edge0.cross(c0)) >= 0
        inside1 = self.plane.normal_directions.dot(edge1.cross(c1)) >= 0
        inside2 = self.plane.normal_directions.dot(edge2.cross(c2)) >= 0
        inside_mask = (inside0 + inside1 + inside2) == 3
        intersection.intersected = intersection.intersected & inside_mask
        intersection.colors_at_intersection = self.get_color_at_point(intersection.intersection_points)
        return intersection

    def get_intersection_t(self, rays):
        t = self.plane.get_intersection_t(rays)
        p = rays.get_point_on_line(t)
        c0 = p - self.a
        c1 = p - self.b
        c2 = p - self.c
        edge0, edge1, edge2 = self.b - self.a, self.c - self.b, self.a - self.c
        inside0 = self.plane.normal_directions.dot(edge0.cross(c0)) >= 0
        inside1 = self.plane.normal_directions.dot(edge1.cross(c1)) >= 0
        inside2 = self.plane.normal_directions.dot(edge2.cross(c2)) >= 0
        inside_mask = (inside0 + inside1 + inside2) == 3
        masked_t = torch.where(inside_mask, t, device_control.get_device_float32_array([t.shape[0]], -1e10))
        return masked_t