from ray_tracing import ray_surface_intersection
import torch
from geometric_primitives import vector
from control import device_control


class Plane:
    # public final Vector3 normalDirection;
    #  public final double d;

    def __init__(self, p1, p2, p3):
        v1 = p1.vector_to_point(p2)
        v2 = p2.vector_to_point(p3)
        self.normal_directions = v1.cross(v2).unit_vectors()
        a, b, c = self.normal_directions._vec[0, 0], self.normal_directions._vec[0, 1], self.normal_directions._vec[0, 2]
        d = -(a * p1._vec[0, 0] + b * p1._vec[0, 1] + c * p1._vec[0, 2])
        self.a, self.b, self.c, self.d = a, b, c, d

    def are_parallel_to(self, v):
        return self.normal_directions.are_perpendiculer_to(v)

    def are_perpendiculer_to(self, v):
        return self.normal_directions.are_parallel_to(v)

    # Does not fill in color
    def find_intersections(self, rays):
        parallel_mask = self.are_parallel_to(rays.directions)
        intersection = ray_surface_intersection.RaySurfaceIntersection()
        # Otherwise there we have intersection
        nom = (-self.d - self.normal_directions.dot(rays.starts))
        denom = self.normal_directions.dot(rays.directions)
        intersection.t = nom / denom
        bad_intersection_mask = (intersection.t < 0) | (intersection.t > 1e15)
        intersection.intersected = 1 - (parallel_mask | bad_intersection_mask)
        intersection.intersection_points = rays.get_point_on_line(intersection.t)
        rev_normal = self.normal_directions.reverse_vector()
        normal_choice_mask = self.normal_directions.dot(rays.directions) < 0
        intersection.intersection_normals = vector.Vector3.where(normal_choice_mask.view(-1, 1), self.normal_directions, rev_normal)
        intersection.reflection_directions = rays.directions.get_reflection_directions(intersection.intersection_normals)
        intersection.incident_rays = rays
        return intersection

    def get_intersection_t(self, rays):
        # If parallel to plane, then no intersection
        parallel_mask = self.are_parallel_to(rays.directions)
        # Otherwise there we have intersection
        nom = (-self.d - self.normal_directions.dot(rays.starts))
        denom = self.normal_directions.dot(rays.directions)
        t = nom / denom
        bad_intersection_mask = (t < 0) | (t > 1e15)
        proper_intersection_mask = 1 - (parallel_mask & bad_intersection_mask)
        n = proper_intersection_mask.shape[0]
        return torch.where(proper_intersection_mask, t, device_control.get_device_float32_array([n], 1e-10))