from geometric_primitives import vector


class Camera:
    __WALK_LENGTH__ = 5
    __ROTATATION_ANGLE__ = 0.0872664


    def __init__(self):
        self.reset_camera()

    def reset_camera(self):
        self.position = vector.Vector3.from_points(80, 35, 0)
        self.look_at = vector.Vector3.from_points(0, 35, 0)
        self.dir_up = vector.Vector3.from_points(0, 1, 0)
        self.look_at = self.position + self.get_forward_vector()

    def get_forward_vector(self):
        return self.position.vector_to_point(self.look_at).unit_vectors()

    def get_right_vector(self):
        r = self.get_forward_vector().cross(self.dir_up)
        r = r.unit_vectors()
        return r

    def get_up_vector(self):
        return self.dir_up

    def get_eye_position(self):
        return self.position

    def walk(self, units):
        d = self.get_forward_vector()
        d = d * units
        self.position += d
        self.look_at += d

    # def rotate_yaw(self, rotationAngle):
    #     pass
    #     f = self.look_at - self.position
    #     forwardDirection = forwardDirection.rotateByAxis(rotationAngle, dirUp)
    #     lookAt = position.addVector(forwardDirection)
    #
    #
    # def rotatePitch(self, rotation_angle):
    #     pass
    #     Vector3 dirRight = getRightVector()
    #     dirUp = dirUp.rotateByAxis(rotationAngle, dirRight)
    #     dirForward = getForwardVector().rotateByAxis(rotationAngle, dirRight)
    #     lookAt = position.addVector(dirForward)
    #
    # def rotateRoll(self, rotationAngle):
    #     pass
    #     dirUp = dirUp.rotateByAxis(rotationAngle, getForwardVector())
    #
    #
    # def straf(self, units):
    #     pass
    #     d = getRightVector().scaleBy(units)
    #     position = position.addVector(d)
    #     lookAt = lookAt.addVector(d)
    #
    # def fly(self, units):
    #     pass
    #     d = dirUp.scaleBy(units)
    #     position = position.addVector(d)
    #     lookAt = lookAt.addVector(d)