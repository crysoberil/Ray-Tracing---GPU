from control import device_control
import numpy as np
import torch
from geometric_primitives import ray
from . import ray_surface_intersection


class RayTracer:
    def __init__(self, shapes, lights, camera, recursive_depth):
        def _extract_shape_statistics(shapes):
            coeffs = []
            for shape in shapes:
                shape_coeff = [shape.ambient_coeff, shape.diffuse_coeff, shape.specular_coeff, shape.specular_exponent, shape.refractive_index, shape.ref_coeff]
                #                     0                     1                    2                      3                         4                    5
                coeffs.append(shape_coeff)
            coeff_tensor = device_control.to_gpu_if_possible(torch.FloatTensor(coeffs))
            return coeff_tensor


        self.shapes = shapes
        self.shape_coefficients = _extract_shape_statistics(shapes)
        self.lights = lights
        self.camera = camera
        self.recursive_depth = recursive_depth

    def _get_initial_rays(self, image_dimension):
        indices = np.indices((image_dimension, image_dimension)).reshape((2, -1)).transpose()
        indices_torch = device_control.to_gpu_if_possible(torch.FloatTensor(indices))
        multiplier = 1.4
        normalized_i = (-indices_torch[:, 0] / image_dimension) + 0.5
        normalized_j = (indices_torch[:, 1] / image_dimension) - 0.5
        normalized_i *= multiplier
        normalized_j *= multiplier
        ray_directions = self.camera.get_right_vector().scale_by(normalized_j) + self.camera.get_up_vector().scale_by(normalized_i) + self.camera.get_forward_vector()
        return ray.Ray(self.camera.position, ray_directions)

    def trace_rays(self, image_dimension):
        init_rays = self._get_initial_rays(image_dimension)
        traced_colors = self._trace_view_rays_out(init_rays, self.recursive_depth)
        traced_colors = traced_colors.view(image_dimension, image_dimension, -1)
        return device_control.extract_value_from_tensor(traced_colors)

    def _trace_view_rays_out(self, rays, recursive_depth):
        n = rays.directions.shape[0]
        if recursive_depth == 0:
            return device_control.get_device_float32_array([n, 3], 0)

        closest_intersections = ray_surface_intersection.RaySurfaceIntersection()
        colors = device_control.get_device_float32_array([n, 3], 0)

        closest_shapes = self._get_closest_ray_intersections(rays, closest_intersections)

        # Mask out the non intersected ones
        further_indices = torch.nonzero(closest_intersections.intersected).view(-1)
        if further_indices.shape[0] > 0:
            closest_intersections = closest_intersections.mask(further_indices)
            closest_shapes = closest_shapes[further_indices]

            # colors_at_intersections = closest_shapes.get_color_at_point(closest_intersections.intersection_points)
            colors_at_intersections = closest_intersections.colors_at_intersection
            colors_from_light = self._get_color_contribution_from_lights(closest_shapes, closest_intersections, colors_at_intersections)
            # Contribution from reflection
            reflected_view_ray = ray.Ray(closest_intersections.intersection_points, closest_intersections.reflection_directions)
            reflected_view_ray.advance_by_epsilon()
            reflection_scale = self.shape_coefficients[closest_shapes, 5]
            reflection_contribution = self._trace_view_rays_out(reflected_view_ray, recursive_depth - 1) * reflection_scale.view(-1, 1)

            # if (!closestShape.isRefractable()) // not triangle
            #     return colorFromLight.add(reflectionContribution);
            #
            # // Must consider total internal reflection
            # double[] REFRACTION_ALPHA = new double[1];
            # Color3 refractionContribution = getRefractionColor(closestIntersection, recursionDepthLeft - 1, closestShape,
            #         REFRACTION_ALPHA).scaleBy(closestShape.refCoeff);
            #
            # Color3 finalColor = colorFromLight;
            # finalColor = finalColor.add(refractionContribution.scaleBy(REFRACTION_ALPHA[0]));
            # finalColor = finalColor.add(reflectionContribution.scaleBy(1 - REFRACTION_ALPHA[0]));
            final_colors = colors_from_light + reflection_contribution

            colors[further_indices, :] = final_colors
        return colors


    def _get_closest_ray_intersections(self, rays, best_intersection):
        # Returns closest shape indices
        n = rays.directions.shape[0]
        closest_shapes = device_control.get_device_int64_array([n], 0)
        for i in range(len(self.shapes)):
            new_intersections = self.shapes[i].find_intersections(rays)
            best_intersection.update_intersection_if_required(new_intersections)
            closest_shapes[best_intersection.updated_indices] = i
        return closest_shapes


    def _get_color_contribution_from_lights(self, closest_shapes, closest_intersections, colors_at_intersections):
        # TODO improve performance for pure black colors
        res = colors_at_intersections * self.shape_coefficients[closest_shapes, 0].view(-1, 1)
        n = closest_shapes.shape[0]
        # Contributions from the light sources
        for light_pos in self.lights:
            t_light = closest_intersections.intersection_points.distances_to(light_pos)
            t_light_mask = t_light > 0
            # if (tLight < 0)
            #     continue;

            rays_to_light = ray.Ray(closest_intersections.intersection_points, light_pos - closest_intersections.intersection_points);
            rays_to_light.advance_by_epsilon()
            rays_reach_lights = device_control.get_device_uint8_array([n], 1)
            for shape in self.shapes:
                new_t = shape.get_intersection_t(rays_to_light)
                obstacle = (new_t >= 0) & (new_t < t_light)
                rays_reach_lights = rays_reach_lights & (1 - obstacle)
            v = closest_intersections.incident_rays.directions.reverse_vector().unit_vectors()
            r = rays_to_light.directions.reverse_vector().get_reflection_directions(closest_intersections.intersection_normals)
            color_from_light = self._phong_illumination_color(closest_shapes, colors_at_intersections, closest_intersections.intersection_normals, rays_to_light.directions, r, v)
            res += torch.where((t_light_mask & rays_reach_lights).view(-1, 1), color_from_light, device_control.get_device_float32_array([n, 3], 0.0))
        return res

    def _phong_illumination_color(self, closest_shapes, colors_at_intersections, n, s, r, v):
        count = closest_shapes.shape[0]
        diffuse_scale = torch.max(device_control.get_device_float32_array([count], 0.0), 1.0 / len(self.lights) * self.shape_coefficients[closest_shapes, 1] * s.dot(n))
        diffuse_color = colors_at_intersections * diffuse_scale.view(-1, 1)
        r_dot_v = r.dot(v)
        speculer_scale = (1.0 / len(self.lights) * self.shape_coefficients[closest_shapes, 2] * torch.pow(r_dot_v, self.shape_coefficients[closest_shapes, 3])).view(-1, 1)
        specular_color = device_control.get_device_float32_array([count, 3], 1.0) * speculer_scale
        combined = diffuse_color + specular_color
        return combined

