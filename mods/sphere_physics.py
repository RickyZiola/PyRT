import numpy
from mods.vec3 import vec3
from main import framerate
class Sphere_Rigidbody:
    def __init__(self, obj, mass, gravity = .0098):
      self.obj  =  obj
      self.mass = mass
      self.g = gravity
      self.vel = vec3(0, 0, 0)
    def update(self, scene):
      if(self.collision(scene)[0].components() == (0,0,0)):
        self.vel += vec3(0, -self.g, 0)
      else:
        c = self.collision(scene)
        d = (c[1].r + self.obj.r) - (abs(c[2]) ** (1./2))
        self.obj.c += c[2].norm() * d
        c1 = self.collision(scene)
        if(c1[1] != None):
          self.obj.c -= c[2].norm() * d
          c1[1].c -= c[2].norm() * d
        tval = vec3(-c[0].components()[1], c[0].components()[0], c[0].components()[2])
        tval = tval.norm()
        length = self.vel.dot(tval)
        vcomp_on_tan = tval * length
        try:
          rvel = c[1].rb.vel - self.vel
        except:
          rvel = self.vel
        try:
          c[1].rb
          mv = vec3 (self.mass, c[1].rb.mass, 0).norm()
          self.vel += (rvel - vcomp_on_tan) * mv.components()[1] * .95
          c[1].rb.vel -= (rvel - vcomp_on_tan) * mv.components()[0] * .95
        except:
          self.vel -= (rvel - vcomp_on_tan) * 1.9
        #print(self.vel.components())
        #print(s)
        
        #self.vel.scale(s)
        #print(self.vel)
        #self.vel = self.vel.rotate(self.collision(scene))
      self.obj.c += self.vel*(30/framerate) * .5
    def collision(self, scene):
      for o in [obj for obj in scene if obj != self.obj]:
        if(abs(self.obj.c - o.c) ** (1./2) < self.obj.r + o.r):
          return ((self.obj.c - o.c).norm(), o, self.obj.c - o.c)
      return (vec3(0, 0, 0), None, vec3(0, 0, 0))