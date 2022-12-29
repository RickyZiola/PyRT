import numpy as np
from main import extract
class vec3():
    def __init__(self, x, y, z):
        (self.x, self.y, self.z) = (x, y, z)
    def __mul__(self, other):
        return vec3(self.x * other, self.y * other, self.z * other)
    def __add__(self, other):
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    def __sub__(self, other):
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    def __truediv__(self, other):
        return vec3(self.x / other.x, self.y / other.y, self.z / other.z)
    def dot(self, other):
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)
    def __abs__(self):
        return self.dot(self)
    def scale(self, other):
        self.x *= other.x
        self.y *= other.y
        self.z *= other.z
    def norm(self):
        mag = np.sqrt(abs(self))
        return self * (1.0 / np.where(mag == 0, 1, mag))
    def components(self):
        return (self.x, self.y, self.z)
    def extract(self, cond):
        return vec3(extract(cond, self.x),
                    extract(cond, self.y),
                    extract(cond, self.z))
    def rotate(self, other, degs):
      rotation_degrees = degs
      rotation_radians = np.radians(rotation_degrees)
      rotation_axis = np.array(other.components())

      rotation_vector = rotation_radians * rotation_axis
      rotation = R.Rotation.from_rotvec(rotation_vector)
      rotated_vector = rotation.apply(np.array(self.components()))
      return vec3(rotated_vector[0], rotated_vector[1], rotated_vector[2])

    def place(self, cond):
        r = vec3(np.zeros(cond.shape), np.zeros(cond.shape), np.zeros(cond.shape))
        np.place(r.x, cond, self.x)
        np.place(r.y, cond, self.y)
        np.place(r.z, cond, self.z)
        return r
    def cross_product(self, other):

      x = self.y*other.z - self.z*other.y

      y = self.z*other.x - self.x*other.z

      z = self.x*other.y - self.y*other.x

      return vec3(x, y, z)
rgb = vec3