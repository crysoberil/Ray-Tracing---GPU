import torch
from control import device_control
import math


class Vector3:
    @classmethod
    def from_points(cls, x, y, z):
        vec = device_control.to_gpu_if_possible(torch.FloatTensor([[x, y, z]]))
        return Vector3(vec)

    # Takes in a tensor
    def __init__(self, ref):
        self._vec = ref

    def __add__(self, sec):
        ad = sec if isinstance(sec, torch.Tensor) else sec._vec
        return Vector3(self._vec + ad)

    def __sub__(self, sec):
        sb = sec if isinstance(sec, torch.Tensor) else sec._vec
        return Vector3(self._vec - sb)

    def vector_to_point(self, p):
        return p - self

    def scale_by(self, f):
        if not type(f) in [int, float] and len(f.shape) == 1:
            f = f.view(-1, 1)
        return Vector3(self._vec * f)

    def get_lengths(self):
        squared = self._vec * self._vec
        summed = torch.sum(squared, dim=1)
        norms = torch.sqrt(summed)
        return norms

    def unit_vectors(self, length_threshold=1e-5):
        lengths = self.get_lengths()
        lengths = lengths.view(-1, 1)
        bad_length_mask = lengths < length_threshold
        bad_length_count = device_control.extract_value_from_tensor(torch.sum(bad_length_mask))
        if bad_length_count > 0:
            normalized = torch.where(bad_length_mask, device_control.to_gpu_if_possible(torch.zeros_like(self._vec)), self._vec / lengths)
        else:
            normalized = self._vec / lengths
        return Vector3(normalized)

    def distances_to(self, sec):
        return (sec - self).get_lengths()

    def reverse_vector(self):
        return Vector3(-self._vec)

    def equals(self, sec, distance_threshold=1e-5):
        diff = torch.abs(self._vec - sec._vec)
        bad_dists = device_control.extract_value_from_tensor(torch.sum(diff > distance_threshold))
        return bad_dists == 0

    def dot(self, sec):
        mult = self._vec * sec._vec
        return torch.sum(mult, dim=1)

    def cross(self, sec):
        x = (self._vec[:, 1] * sec._vec[:, 2] - self._vec[:, 2] * sec._vec[:, 1]).view(-1, 1)
        y = (self._vec[:, 2] * sec._vec[:, 0] - self._vec[:, 0] * sec._vec[:, 2]).view(-1, 1)
        z = (self._vec[:, 0] * sec._vec[:, 1] - self._vec[:, 1] * sec._vec[:, 0]).view(-1, 1)
        crossed = torch.cat((x, y, z), dim=1)
        return Vector3(crossed)

    def are_zero_vector(self):
        zeros = device_control.to_gpu_if_possible(torch.zeros_like(self._vec))
        return self.equals(Vector3(zeros))

    def are_parallel_to(self, sec):
        crossed = self.cross(sec)
        return crossed.are_zero_vector()

    def are_perpendiculer_to(self, sec, threshold=1e-5):
        prod = self.dot(sec)
        return torch.abs(prod) < threshold

    def get_reflection_directions(self, normals):
        perpendiculer_mask = self.are_perpendiculer_to(normals)
        # // a - 2(a.n)n
        v2 = normals.scale_by(2 * self.dot(normals))
        ret = (self - v2).unit_vectors()
        return Vector3.where(perpendiculer_mask, self.unit_vectors(), ret)

    def rotate_by_axis(self, theta, axis):
        """
        :param theta: theta in radian
        :param axis:
        :return:
        """
        assert self._vec.shape[0] == 1
        u, v, w = device_control.extract_value_from_tensor(axis).ravel()
        xPrime = u * (u * self._vec[0, 0] + v * self._vec[0, 1] + w * self._vec[0, 2]) * (1.0 - math.cos(theta)) + self._vec[0, 0] * math.cos(theta) + (-w * self._vec[0, 1] + v * self._vec[0, 2]) * math.sin(theta)
        yPrime = v * (u * self._vec[0, 0] + v * self._vec[0, 1] + w * self._vec[0, 2]) * (1.0 - math.cos(theta)) + self._vec[0, 1] * math.cos(theta) + (w * self._vec[0, 0] - u * self._vec[0, 2]) * math.sin(theta)
        zPrime = w * (u * self._vec[0, 0] + v * self._vec[0, 1] + w * self._vec[0, 2]) * (1.0 - math.cos(theta)) + self._vec[0, 2] * math.cos(theta) + (-v * self._vec[0, 0] + u * self._vec[0, 1]) * math.sin(theta)
        return Vector3(xPrime, yPrime, zPrime)

    def mask(self, indices):
        if self._vec.shape[0] == 1:
            masked = self._vec
        else:
            masked = self._vec[indices, :]
        return Vector3(masked)

    @staticmethod
    def where(selector, v1, v2):
        if len(selector.shape) == 1:
            selector = selector.view(-1, 1)
        v = torch.where(selector, v1._vec, v2._vec)
        return Vector3(v)

    def __str__(self):
        return self._vec.__str__()

    @property
    def shape(self):
        return self._vec.shape


    # public static Vector3 read(Scanner in) {
    #     double x = in.nextDouble();
    #     double y = in.nextDouble();
    #     double z = in.nextDouble();
    #     return new Vector3(x, y, z);
    # }
    #
    # @Override
    # public String toString() {
    #     return "V=(" + x + "," + y + "," + z + ")";