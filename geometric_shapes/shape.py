from control import device_control
from utils import parse_utils

class Shape:
    # def __init__(self):
    #     self.ambient_coeff = -1.0
    #     self.diffuse_coeff = -1.0
    #     self.specular_coeff = -1.0
    #     self.specular_exponent = -1.0
    #     self.refractive_index = -1.0
    #     self.ref_coeff = -1.0
    #     self.color = device_control.get_device_float32_array([3], 0.0)
    def __init__(self):
        self.refractive_index = 0.0

    def load_default_property(self, property):
        if property[0] == "ambCoeff":
            self.ambient_coeff = float(property[1])
        elif property[0] == "difCoeff":
            self.diffuse_coeff = float(property[1])
        elif property[0] == "refCoeff":
            self.ref_coeff = float(property[1])
        elif property[0] == "specCoeff":
            self.specular_coeff = float(property[1])
        elif property[0] == "specExp":
            self.specular_exponent = float(property[1])
        elif property[0] == "refractiveIndex":
            self.refractive_index = float(property[1])
        elif property[0] == "color":
            self.color = parse_utils.parse_tensor_from_string_toks(property[1: ]).view(1, -1)

    def find_intersections(ray):
        pass

    def get_intersection_t(ray):
        pass

    def get_color_at_point(p):
        pass

    def is_refractable(self):
        return False