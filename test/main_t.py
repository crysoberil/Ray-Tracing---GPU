from geometric_shapes import checkerboard
from geometric_primitives import vector
from control import device_control
import torch
from ray_tracing import trace_rays
from main_package import camera, display_utils


c1 = device_control.to_gpu_if_possible(torch.FloatTensor([1.0, 1.0, 1.0]))
c2 = device_control.to_gpu_if_possible(torch.FloatTensor([0.0, 0.0, 0.0]))

ckbd = checkerboard.CheckerBoard(5, c1, c2, .4, .1, .4, .1, 3.)
shapes = [ckbd]

l1 = vector.Vector3(device_control.to_gpu_if_possible(torch.FloatTensor([70.0, 70.0, 70.0])))
l2 = vector.Vector3(device_control.to_gpu_if_possible(torch.FloatTensor([-70.0, 1.0, -70.0])))

lights = [l1, l2]

camera = camera.Camera()
rt = trace_rays.RayTracer(shapes, lights, camera, 1)
traced = rt.trace_rays(300)
display_utils.display_numpy_image(traced)