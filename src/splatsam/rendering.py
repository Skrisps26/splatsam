"""Configure software-only rendering before importing MuJoCo."""
import ctypes.util
import os
os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
os.environ.setdefault('GALLIUM_DRIVER', 'llvmpipe')
os.environ.setdefault('MUJOCO_GL', 'osmesa' if ctypes.util.find_library('OSMesa') else 'egl')

def renderer(model, height=480, width=640):
    import mujoco
    return mujoco.Renderer(model, height=height, width=width)
