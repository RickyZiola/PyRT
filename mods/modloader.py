import os
import importlib

blacklisted = ["modloader.py", "vec3.py"]

files = [f[:-3] for f in os.listdir("mods") if f[-3:] == ".py" and f not in blacklisted] 
print(files)
for lib in files:
  importlib.import_module("mods." + lib)
print(files)