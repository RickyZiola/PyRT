from mods.vec3 import vec3, rgb
import numpy
from main import FARAWAY
from mods.sphere_rendering import Sphere
import main
import trimesh
import numpy as np
import os
os.system("python3 -m pip install rtree")
def ray_intersects_mesh(origin, direction, mesh):
    # Convert the origin and direction lists to arrays
    ray_origin = np.array(origin, 'float64')
    ray_direction = np.array(direction, 'float64')
    #print(ray_direction)
    # Use Trimesh to intersect the ray with the mesh
    origins, directions = [],[]
    for d in zip(ray_direction[0], ray_direction[1], ray_direction[2]):
      directions.append(d)
      origins.append(ray_origin)
    intersections = mesh.ray.intersects_id(origins, directions, True, False)
    if len(intersections) > 0:
        # Intersection found
        return abs(intersections[0][2] - origin)
    else:
        # No intersection found
        return FARAWAY

class Mesh:
  def __init__(self, fname, c):
    self.v, self.f = load_mesh("mesh.obj")
    self.diffuse = c

  def intersect(self, o, d):
    int = ray_intersects_mesh(o.components(), d.components(), trimesh.Trimesh(self.v, self.f))
    print(int)
    return int

  def diffusecolor(self,M):
    return self.diffuse

  def light(self, O, D, d, scene, bounce, sun=rgb(.8, .8, 0)):
    t = self.intersect(O, D)
    return self.diffuse/rgb(t,t,t)


def load_mesh(filename):
    """Load a mesh from an obj file.

    Args:
        filename: The path to the obj file.

    Returns:
        A tuple containing two lists:
            The first list contains the vertexes of the mesh.
            The second list contains the faces of the mesh,
            represented as tuples of indices into the vertex list.
    """
    vertexes = []
    faces = []

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v '):
                # Parse the vertex coordinates from the line.
                vertex = tuple(map(float, line.split()[1:4]))
                vertexes.append(vertex)
            elif line.startswith('f '):
                # Parse the face vertex indices from the line.
                face = tuple(map(int, line.split()[1:4]))
                faces.append(face)
    print(vertexes, faces)
    return vertexes, faces
main.scene = [
  Sphere(vec3(0, 0, 0), .6, rgb(0, 0, 1)),
  Mesh("mesh.obj", rgb(1, 0, 0)),
]
main.init()
main.run()