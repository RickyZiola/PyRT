from PIL import Image
from functools import reduce
import numpy as np
import time
import numbers
import cv2
import scipy.spatial.transform as R
import math

# import only system from os
from os import system, name
 
# import sleep to show output for some time period
from time import sleep
 
# define our clear function
def clear():
 
    # for windows
    if name == 'nt':
        _ = system('cls')
 
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')

def extract(cond, x):
    if isinstance(x, numbers.Number):
        return x
    else:
        return np.extract(cond, x)

class vec3():
    def __init__(self, x, y, z):
        (self.x, self.y, self.z) = (x, y, z)
    def __mul__(self, other):
        return vec3(self.x * other, self.y * other, self.z * other)
    def __add__(self, other):
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    def __sub__(self, other):
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    def __div__(self, other):
        return vec3(self.x / other, self.y / other, self.z / other)
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
rgb = vec3
res_low   = ( 100,   57)
res_240p  = ( 426,  240)
res_480p  = ( 854,  480)
res_720p  = (1280,  720)
res_1080p = (1920, 1080)
res_1440p = (2650, 1440)
res_2160p = (3840, 2160)
(w, h) = res_480p        # Screen size
L = vec3(5, 5, -10)        # Point light position
E = vec3(0, 0.35, -1)     # Eye position
FARAWAY = 1.0e39       # an implausibly huge distance


frameSize = (w,h)
vid_len = 5
framerate = 30
out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), framerate, frameSize)

def raytrace(O, D, scene, bounce = 0):
    # O is the ray origin, D is the normalized ray direction
    # scene is a list of Sphere objects (see below)
    # bounce is the number of the bounce, starting at zero for camera rays

    distances = [s.intersect(O, D) for s in scene]
    nearest = reduce(np.minimum, distances)
    #color = rgb(135/255, 206/255, 235/255)
    color = rgb(0,0,0)
    for (s, d) in zip(scene, distances):
        hit = (nearest != FARAWAY) & (d == nearest)
        if np.any(hit):
            dc = extract(hit, d)
            Oc = O.extract(hit)
            Dc = D.extract(hit)
            cc = s.light(Oc, Dc, dc, scene, bounce)
            color += cc.place(hit)
    return color
    #if(nearest != FARAWAY):
      #return color
    #else:
      #return rgb(135/255, 206/255, 235/255)

class Rigidbody:
    def __init__(self, obj, mass):
      self.obj  =  obj
      self.mass = mass
      self.vel = vec3(0, 0, 0)
    def update(self, scene):
      if(self.collision(scene)[0].components() == (0,0,0)):
        self.vel += vec3(0, -.0098, 0)
      else:
        c = self.collision(scene)
        d = (c[1].r + self.obj.r) - (abs(c[2]) ** (1./2))
        self.obj.c += c[2].norm() * d
        tval = vec3(-c[0].components()[1], c[0].components()[0], c[0].components()[2])
        tval = tval.norm()
        length = self.vel.dot(tval)
        vcomp_on_tan = tval * length
        try:
          rvel = c[1].rb.vel - self.vel
        except:
          rvel = self.vel
        if(c[1].rb == None):
          self.vel -= (rvel - vcomp_on_tan) * 1.9
        else:
          mv = vec3 (self.mass, c[1].rb.mass, 0).norm()
          self.vel += (rvel - vcomp_on_tan) * mv.components()[1] * 1.9
          c[1].rb.vel -= (rvel - vcomp_on_tan) * mv.components()[0] * 1.9
        #print(self.vel.components())
        #print(s)
        
        #self.vel.scale(s)
        #print(self.vel)
        #self.vel = self.vel.rotate(self.collision(scene))
      self.obj.c += self.vel*(30/framerate)
    def collision(self, scene):
      for o in [obj for obj in scene if obj != self.obj]:
        if(abs(self.obj.c - o.c) ** (1./2) < self.obj.r + o.r):
          return ((self.obj.c - o.c).norm(), o, self.obj.c - o.c)
      return (vec3(0, 0, 0), None, vec3(0, 0, 0))
class Sphere:
    def __init__(self, center, r, diffuse, mirror = 0.5):
        self.c = center
        self.r = r
        self.diffuse = diffuse
        self.mirror = mirror
        self.rb = None

    def intersect(self, O, D):
        b = 2 * D.dot(O - self.c)
        c = abs(self.c) + abs(O) - 2 * self.c.dot(O) - (self.r * self.r)
        disc = (b ** 2) - (4 * c)
        sq = np.sqrt(np.maximum(0, disc))
        h0 = (-b - sq) / 2
        h1 = (-b + sq) / 2
        h = np.where((h0 > 0) & (h0 < h1), h0, h1)
        pred = (disc > 0) & (h > 0)
        return np.where(pred, h, FARAWAY)

    def diffusecolor(self, M):
        return self.diffuse

    def light(self, O, D, d, scene, bounce, sun=rgb(.8, .8, 0)):
        M = (O + D * d)                         # intersection point
        N = (M - self.c) * (1. / self.r)        # normal
        toL = (L - M).norm()                    # direction to light
        toO = (E - M).norm()                    # direction to ray origin
        nudged = M + N * .0001                  # M nudged to avoid itself

        # Shadow: find if the point is shadowed or not.
        # This amounts to finding out if M can see the light
        light_distances = [s.intersect(nudged, toL) for s in scene]
        light_nearest = reduce(np.minimum, light_distances)
        seelight = light_distances[scene.index(self)] == light_nearest

        # Ambient
        color = rgb(0.05, 0.05, 0.05)

        # Lambert shading (diffuse)
        lv = np.maximum(N.dot(toL), 0)
        color += self.diffusecolor(M) * lv * seelight

        # Reflection
        if bounce < 2:
            rayD = (D - N * 2 * D.dot(N)).norm()
            color += raytrace(nudged, rayD, scene, bounce + 1) * self.mirror

        # Blinn-Phong shading (specular)
        phong = N.dot((toL + toO).norm())
        color += sun * np.power(np.clip(phong, 0, 1), 50) * seelight
        return color

class CheckeredSphere(Sphere):
    def diffusecolor(self, M):
        checker = ((M.x * 2).astype(int) % 2) == ((M.z * 2).astype(int) % 2)
        return self.diffuse * checker

class MirrorSphere(Sphere):
    def light(self, O, D, d, scene, bounce, sun=rgb(.8, .8, 0)):
        M = (O + D * d)                         # intersection point
        N = (M - self.c) * (1. / self.r)        # normal
        toL = (L - M).norm()                    # direction to light
        toO = (E - M).norm()                    # direction to ray origin
        nudged = M + N * .0001                  # M nudged to avoid itself

        # Shadow: find if the point is shadowed or not.
        # This amounts to finding out if M can see the light
        light_distances = [s.intersect(nudged, toL) for s in scene]
        light_nearest = reduce(np.minimum, light_distances)
        seelight = light_distances[scene.index(self)] == light_nearest

        # Ambient
        color = rgb(0.05, 0.05, 0.05)

        # Lambert shading (diffuse)
        lv = np.maximum(N.dot(toL), 0)
        color += rgb(1,1,1) * lv * seelight

        # Reflection
        if bounce < 2:
            rayD = (D - N * 2 * D.dot(N)).norm()
            color += raytrace(nudged, rayD, scene, bounce + 1) * self.mirror

        # Blinn-Phong shading (specular)
        phong = N.dot((toL + toO).norm())
        color += sun * np.power(np.clip(phong, 0, 1), 50) * seelight
        return color
scene = [
    #Sphere(vec3(-5, .1, .5), .6, rgb(0, 0, 1), 1),
    Sphere(vec3(-.75, 1.5, 2), .6, rgb(1, 0, 0), 1),
    Sphere(vec3(.75, 1.5, 2), .6, rgb(1, 1, 1), 1),
    CheckeredSphere(vec3(0,-99999.5, 0), 99999, rgb(.75, .75, .75), 0),
    
    #MirrorSphere(vec3(0, 1, 99999+2), 99999, rgb(0, 1, 0), 1)
    ]
rb1 = Rigidbody(scene[0], 1)
scene[0].rb = rb1
rb2 = Rigidbody(scene[1], 1)
scene[1].rb = rb2
d = [-.05*(30/framerate), .05*(30/framerate)]
r = float(w) / h
al = []
for f in range(vid_len*framerate):
  #scene[0].c += vec3(0.05*(30/framerate), 0, 0)
  #if(scene[0].c.x > 5):
    #scene[0].c.x = -5
  rb1.update(scene)
  rb2.update(scene)
  print(rb1.vel.components())
  print(rb2.vel.components())
  """
  for i in [1,2]:
    scene[i].c.y += d[i-1]
    if(abs(scene[i].c - scene[3].c) ** (1./2) < scene[i].r + scene[3].r):
      d[i-1] = -d[i-1]
    if(scene[i].c.y > 2):
      d[i-1] = -d[i-1]"""
  # Screen coordinates: x0, y0, x1, y1.
  S = (-1, 1 / r + .25, 1, -1 / r + .25)
  x = np.tile(np.linspace(S[0], S[2], w), h)
  y = np.repeat(np.linspace(S[1], S[3], h), w)
  
  t0 = time.time()
  Q = vec3(x, y, 0)
  color = raytrace(E, (Q - E).norm(), scene)
  #print ("Took", time.time() - t0)
  
  lrgb = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((h, w))).astype(np.uint8), "L") for c in color.components()]
  Image.merge("RGB", lrgb).save("rt3.png")
  img = cv2.imread("rt3.png")
  out.write(img)
  al.append(time.time() - t0)
  s_left = int(np.average(al) * (vid_len*framerate-f))
  clear()
  print(f"\r{f+1} out of {vid_len*framerate} frames rendered, {int((f+1)/(vid_len*framerate)*10000)/100}%  Estimated time remaining: {int(np.floor(s_left / 60))} minutes, {s_left % 60} seconds.      ")
  

out.release()
