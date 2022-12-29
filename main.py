from PIL import Image
from functools import reduce
import numpy as np
import time
import numbers
import cv2
import scipy.spatial.transform as R
import math
import numpy as np
if __name__ == "__main__":
  from mods.vec3 import vec3, rgb
  import mods.modloader

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

res_low   = ( 100,   57)
res_240p  = ( 426,  240)
res_480p  = ( 854,  480)
res_720p  = (1280,  720)
res_1080p = (1920, 1080)
res_1440p = (2650, 1440)
res_2160p = (3840, 2160)
framerate = 30
FARAWAY = 1.0e39 
def init():
  global L, E, framsSize, vid_len, w, h, out, vec3, rgb
  from mods.vec3 import vec3, rgb
  (w, h) = res_720p        # Screen size
  L = vec3(5, 5, -10)        # Point light position
  E = vec3(0, 0.35, -1)     # Eye position
        # an implausibly huge distance


  frameSize = (w,h)
  vid_len = 5
  out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), framerate, frameSize)

def raytrace(O, D, scene, bounce = 0):
    # O is the ray origin, D is the normalized ray direction
    # scene is a list of Sphere objects (see below)
    # bounce is the number of the bounce, starting at zero for camera rays

    distances = [s.intersect(O, D) for s in scene]
    nearest = reduce(np.minimum, distances)
    #color = rgb(135/255, 206/255, 235/255)
    #color = np.full((w,h), rgb(0,0,0))
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

def run():
  """rb1 = Rigidbody(scene[0], 100)
  rb1.vel = vec3(.02, -.05, 0)
  scene[0].rb = rb1
  rb2 = Rigidbody(scene[1], 1)
  scene[1].rb = rb2"""
  #rb3 = Rigidbody(scene[2], 4.19e15, 0)
  #scene[2].rb = rb3
  d = [-.05*(30/framerate), .05*(30/framerate)]
  r = float(w) / h
  al = []

  for f in range(vid_len*framerate):
    #scene[0].c += vec3(0.05*(30/framerate), 0, 0)
    #if(scene[0].c.x > 5):
      #scene[0].c.x = -5
    #rb1.update(scene)
    #rb2.update(scene)
    #rb3.update(scene)
    #print(rb1.vel.components())
    #print(rb2.vel.components())
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
    print(color)    
    lrgb = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((h, w))).astype(np.uint8), "L") for c in color.components()]
    Image.merge("RGB", lrgb).save("rt3.png")
    img = cv2.imread("rt3.png")
    out.write(img)
    al.append(time.time() - t0)
    s_left = int(np.average(al) * (vid_len*framerate-f))
    clear()
    print(f"\r{f+1} out of {vid_len*framerate} frames rendered, {int((f+1)/(vid_len*framerate)*10000)/100}%  Estimated time remaining: {int(np.floor(s_left / 60))} minutes, {s_left % 60} seconds.      ")
    
  
  out.release()
