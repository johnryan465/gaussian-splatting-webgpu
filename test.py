#%%
import math
from matplotlib import pyplot as plt
from reference.camera import Camera
from reference.loadfile import Gaussians
from reference.render import Renderer
import numpy as np

fy = 1156.2802734375
fx = 1163.2547607421875
imageHeight = 1091
imameWidth = 1957
fovy=2*math.atan(imageHeight/(2*fy))
fovx=2*math.atan(imameWidth/(2*fx))

print(fovx, fovy)
renderer = Renderer()
camera = Camera(
    position=np.array([ 0,0,0 ]),
    rotation=np.array(np.eye(3)),
    fovx=fovx,
    fovy=fovy,
    near=0.02,
    far=1100.0
)
print(camera.world_to_screen)
# %%
def is_inside_frustum(world_space: np.ndarray) -> bool:
    """Check if a point is inside a view frustum"""
    ndc = camera.world_to_ndc(world_space)
    cam_space = camera.world_to_view(world_space)

    res = -1 <= ndc[0] <= 1 and -1 <= ndc[1] <= 1 and camera.near <= cam_space[2] <= camera.far

    return res
point = np.array([0,0,1100])
camera.is_inside_frustum(point)
camera.world_to_ndc(point)

# %%
