import torch
from geometric_primitives import ray, vector
from control import device_control


class RaySurfaceIntersection:
    def __init__(self):
        self.intersected = None

    def update_intersection_if_required(self, sec):
        n = sec.intersected.shape[0]
        if self.intersected is None:
            self.incident_rays = sec.incident_rays
            self.intersected = sec.intersected
            self.t = sec.t
            self.intersection_points = sec.intersection_points
            self.intersection_normals = sec.intersection_normals
            self.reflection_directions = sec.reflection_directions
            self.colors_at_intersection = sec.colors_at_intersection
            self.updated_indices = torch.nonzero(device_control.get_device_int32_array([n], 1)).view(-1)
        else:
            # TODO check if correct
            choose_first = self.intersected & ((1 - sec.intersected) | (sec.intersected & (self.t < sec.t)))
            self.intersected = torch.where(choose_first, self.intersected, sec.intersected)
            self.t = torch.where(choose_first, self.t, sec.t)
            self.intersection_points = vector.Vector3.where(choose_first, self.intersection_points, sec.intersection_points)
            self.intersection_normals = vector.Vector3.where(choose_first, self.intersection_normals, sec.intersection_normals)
            self.reflection_directions = vector.Vector3.where(choose_first, self.reflection_directions, sec.reflection_directions)
            self.incident_rays = ray.Ray.where(choose_first, self.incident_rays, sec.incident_rays)
            self.colors_at_intersection = torch.where(choose_first.view(-1, 1), self.colors_at_intersection, sec.colors_at_intersection)
            self.updated_indices = torch.nonzero(1 - choose_first).view(-1)

    def mask(self, indices):
        masked = RaySurfaceIntersection()
        masked.incident_rays = self.incident_rays.mask(indices)
        masked.intersected = self.intersected[indices]
        masked.t = self.t[indices]
        masked.intersection_points = self.intersection_points.mask(indices)
        masked.intersection_normals = self.intersection_normals.mask(indices)
        masked.reflection_directions = self.reflection_directions.mask(indices)
        masked.colors_at_intersection = self.colors_at_intersection[indices]
        # del masked.updated_indices
        # masked.updated_indices = self.updated_indices[indices]
        return masked

    # public boolean isCloserIntersectionThan(RaySurfaceIntersection sec) {
	# 	return intersected && (!sec.intersected || (sec.intersected && t < sec.t));
	# }
    #
	# public void morphTo(RaySurfaceIntersection sec) {
	# 	incidentRay = sec.incidentRay;
	# 	intersected = sec.intersected;
	# 	t = sec.t;
	# 	intersectionPoint = sec.intersectionPoint;
	# 	intersectionNormal = sec.intersectionNormal;
	# 	reflectionDirection = sec.reflectionDirection;