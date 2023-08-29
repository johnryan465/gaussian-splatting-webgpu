#%%
import math
from matplotlib import pyplot as plt
from reference.camera import Camera
from reference.loadfile import Gaussians
from reference.render import Renderer
import numpy as np

with open("point_cloud.ply", "rb") as f:
    bytes_read = f.read()
gaussians = Gaussians(bytes_read)

#%%
fy = 1156.2802734375
fx = 1163.2547607421875
imageHeight = 1091
imameWidth = 1957
fovy=360.0*math.atan(imageHeight/(2*fy))/math.pi
fovx=360.0*math.atan(imameWidth/(2*fx))/math.pi
print(fovx, fovy)
#%%
fovx = np.deg2rad(80)
fovy = np.deg2rad(50)
print(fovx, fovy)
renderer = Renderer()
camera = Camera(
    position=np.array([ 1.0722368955612183, 0.6061256527900696, -2.6365013122558594 ]),
    rotation=np.array([ [ 0.8878934979438782, -0.07130444049835205, 0.45448988676071167 ], [ -0.07130444049835205, 0.45448988676071167, 0.00947901327162981 ], [ 0.45448988676071167, 0.00947901327162981, 0.9905414581298828 ] ]),
    fovx=fovx,
    fovy=fovy,
    near=0.02,
    far=1100.0
)
img = renderer.render(50, 80, camera, gaussians)
# %%
from PIL import Image
img = img * 255
img = img.astype(np.uint8)
im = Image.fromarray(img)
im.save("out.png")
# %%
img
# %%
