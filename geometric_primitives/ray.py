from . import vector

class Ray:
    __RAY_ADVANCE_EPSILON = 1e-3;

    def __init__(self, starts, directions):
        self.starts = starts
        self.directions = directions.unit_vectors()

    # public Ray(Point3 start, Point3 toPoint) {
    #     if (start.equals(toPoint)) {
    #         double scale = 1e-4;
    #         double dx = (rand.nextDouble() - 0.5) * scale;
    #         double dy = (rand.nextDouble() - 0.5) * scale;
    #         double dz = (rand.nextDouble() - 0.5) * scale;
    #         start = new Point3(start.x + dx, start.y + dy, start.z + dz);
    #     }
    #
    #     this.start = start;
    #     this.direction = start.vectorTo(toPoint).unitVector();
    # }

    def advance_by_epsilon(self):
        advance_by = self.directions.scale_by(Ray.__RAY_ADVANCE_EPSILON)
        self.starts = self.starts + advance_by

    def get_point_on_line(self, t):
        return self.starts + self.directions.scale_by(t)

    def mask(self, indices):
        starts = self.starts.mask(indices)
        directions = self.directions.mask(indices)
        return Ray(starts, directions)

    @staticmethod
    def where(selector, r1, r2):
        s = vector.Vector3.where(selector, r1.starts, r2.starts)
        d = vector.Vector3.where(selector, r1.directions, r2.directions)
        return Ray(s, d)

    # @Override
    # public String toString() {
    #     return "[" + start.toString() + ", " + direction.toString() + "]";
    # }