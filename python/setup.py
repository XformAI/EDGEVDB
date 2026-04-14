"""
Minimal setup.py — main configuration lives in pyproject.toml.
This file exists only to log native library discovery during build.
"""
from setuptools import setup
import os
import glob

lib_base = os.path.join("edgevdb", "lib")
for plat, ext in [("linux", "*.so"), ("darwin", "*.dylib"), ("windows", "*.dll")]:
    found = glob.glob(os.path.join(lib_base, plat, ext))
    tag = "OK" if found else "MISSING"
    print(f"[setup.py] {plat}: {tag} {found}")

setup()
