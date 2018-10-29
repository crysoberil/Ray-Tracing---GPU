import torch
from . import shape
from control import device_control
from geometric_primitives import plane, vector
from utils import parse_utils


class CheckerBoard(shape.Shape):
    @staticmethod
    def parse_from_description(description_list):
        checkerboard = CheckerBoard()
        checkerboard.cell_width = 20
        p1 = vector.Vector3.from_points(1, 0, 1)
        p2 = vector.Vector3.from_points(-1, 0, 1)
        p3 = vector.Vector3.from_points(5, 0, 7)
        checkerboard.plane = plane.Plane(p1, p2, p3)
        for description in description_list:
            if description[0] == "colorOne":
                checkerboard.color1 = parse_utils.parse_tensor_from_string_toks(description[1: ])
            elif description[0] == "colorTwo":
                checkerboard.color2 = parse_utils.parse_tensor_from_string_toks(description[1: ])
            else:
                checkerboard.load_default_property(description)
        return checkerboard

    def __init__(self):
        super(CheckerBoard, self).__init__()

    # def __init__(self, cell_width, color1, color2, amb_coeff, diff_coeff, ref_coeff, spec_coeff, spec_exp, refractive_index=0.0):
    #     p1 = vector.Vector3.from_points(1, 0, 1)
    #     p2 = vector.Vector3.from_points(-1, 0, 1)
    #     p3 = vector.Vector3.from_points(5, 0, 7)
    #     self.plane = plane.Plane(p1, p2, p3)
    #     self.cell_width = cell_width
    #     self.color1 = color1
    #     self.color2 = color2
    #     self.ambient_coeff = amb_coeff
    #     self.diffuse_coeff = diff_coeff
    #     self.ref_coeff = ref_coeff
    #     self.specular_coeff = spec_coeff
    #     self.specular_exponent = spec_exp
    #     self.refractive_index = refractive_index

    def _rounded_number(self, n):
        target_cast_type = torch.cuda.IntTensor if device_control._device is not None else torch.IntTensor
        non_negative = n > -1e-10
        v1 = n.type(target_cast_type)
        v2 = (n - 1).type(target_cast_type)
        rounded = torch.where(non_negative, v1, v2)
        return rounded

    def get_color_at_point(self, p):
        d_i = p._vec[:, 0] / self.cell_width
        d_j = p._vec[:, 2] / self.cell_width
        i = self._rounded_number(d_i)
        j = self._rounded_number(d_j)
        mask = (i & 1) == (j & 1)
        col = torch.where(mask.view(-1, 1), self.color1.view(1, -1), self.color2.view(1, -1))
        return col

    def find_intersections(self, rays):
        intersections = self.plane.find_intersections(rays)
        intersections.colors_at_intersection = self.get_color_at_point(intersections.intersection_points)
        return intersections

    def get_intersection_t(self, rays):
        return self.plane.get_intersection_t(rays)