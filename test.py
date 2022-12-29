import trimesh
import numpy as np

mesh = trimesh.ray.ray_triangle.RayMeshIntersector(trimesh.load_mesh("mesh.obj"))

if(mesh.intersects_any([(0,0,0)], [(0,0,1)])):
  print("INTERSECT")
else:
  print("NO INTERSECT")