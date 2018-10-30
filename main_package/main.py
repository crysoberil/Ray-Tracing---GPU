import argparse
import torch
from control import device_control
from geometric_primitives import vector
from ray_tracing import trace_rays
from main_package import display_utils
from main_package import camera as camera_module
from geometric_shapes import checkerboard, sphere, triangle


def parse_dsl_file(file_path):
    def parse_shape(description):
        object_type = description[0][0]
        description = description[1: ]
        if object_type == "CHECKERBOARD":
            return checkerboard.CheckerBoard.parse_from_description(description)
        if object_type == "SPHERE":
            return sphere.Sphere.parse_from_description(description)
        if object_type == "TRIANGLE":
            return triangle.Triangle.parse_from_description(description)

    with open(file_path, 'r') as f_in:
        shapes = []
        lights = []

        for line in f_in:
            line = line.strip()
            if len(line) == 0:
                continue
            line_toks = line.split()
            if line_toks[0] == "recDepth":
                recursive_depth = int(line_toks[1])
            elif line_toks[0] == "pixels":
                image_dimension = int(line_toks[1])
            elif line_toks[0] == "light":
                light_position = torch.FloatTensor([[float(elm) for elm in line_toks[1: ]]])
                light_position = vector.Vector3(device_control.to_gpu_if_possible(light_position))
                lights.append(light_position)
            elif line_toks[0] == "objStart":
                current_shape_desciprtion = [[line_toks[1]]]
            elif line_toks[0] == "objEnd":
                parsed_shape = parse_shape(current_shape_desciprtion)
                if parsed_shape is not None:
                    shapes.append(parsed_shape)
            else:
                current_shape_desciprtion.append(line_toks)

    return shapes, lights, recursive_depth, image_dimension


def _main_cli(args):
    shapes, lights, recursive_depth, image_dimension = parse_dsl_file(args.sdl_path)
    camera = camera_module.Camera()
    ray_tracer = trace_rays.RayTracer(shapes, lights, camera, recursive_depth)
    traced = ray_tracer.trace_rays(image_dimension)
    display_utils.display_numpy_image(traced)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("sdl_path", type=str, help="Path to the scene description file.", default="../resources/scene_description.txt")
    args = parser.parse_args()
    _main_cli(args)