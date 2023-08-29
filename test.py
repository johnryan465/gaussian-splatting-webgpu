#%%
%reload_ext autoreload
%autoreload 2
import math
from matplotlib import pyplot as plt
from tqdm import tqdm
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

near = 1
far = 10.0
fov = 60 # Field of view in degrees


print(fovx, fovy)
renderer = Renderer()
camera = Camera(
    position=np.array([ 1.0722368955612183, 0.6061256527900696, -2.6365013122558594 ]),
    rotation=np.array([ [ 0.8878934979438782, -0.07130444049835205, 0.45448988676071167 ], [ -0.07130444049835205, 0.45448988676071167, 0.00947901327162981 ], [ 0.45448988676071167, 0.00947901327162981, 0.9905414581298828 ] ]),
    fovx=np.deg2rad(fov),
    fovy=np.deg2rad(fov),
    near=near,
    far=far
)
print(camera.world_to_screen)
# %%
"""Render a scene from a camera"""
with open("point_cloud.ply", "rb") as f:
    bytes_read = f.read()
gaussians = Gaussians(bytes_read)
width = 50
height = 50

#%%
# Filter gaussian points that are outside the view frustum
filtered_gaussians = []
for gaussian in gaussians.gaussians:
    pos = gaussian.position
    if camera.is_inside_frustum(pos):
        # camera_space_pos_4d = camera.camera_matrix @ np.concatenate((pos, np.array([1])), axis=0)
        gaussian.camera_space_pos = camera.world_to_ndc(pos)
        filtered_gaussians.append(gaussian)

# %%
# Sort gaussians by distance to camera
sorted_gaussians = sorted(filtered_gaussians, key=lambda gaussian: gaussian.camera_space_pos[2])

# Render gaussians
img = np.zeros((width, height, 3))
alpha = np.ones((width, height))
print(camera.world_to_screen)


pixels = [(x, y) for x in range(width) for y in range(height)]
count = 0

#%%
mx = []
my = []
mz = []

bx = []
by = []
bz = []

count = 0
for gaussian in tqdm(gaussians.gaussians):
    pos = gaussian.position
    res = camera.world_to_view(pos)
    if(camera.is_inside_frustum(pos)):
        bx.append(res[0])
        by.append(res[1])
        bz.append(res[2])
    else:
        mx.append(res[0])
        my.append(res[1])
        mz.append(res[2])
    count += 1
    if count > 5000:
        break
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

# Creating plot
ax.scatter3D(mx, my, mz, color = "green")
plt.title("simple 3D scatter plot")
plt.show()
# %%
%matplotlib widget
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Set up figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set camera parameters
aspect_ratio = 1

# Calculate width and height of near and far plane 
hnear = 2 * np.tan(np.deg2rad(fov / 2)) * near
wnear = hnear * aspect_ratio
hfar = 2 * np.tan(np.deg2rad(fov / 2)) * far
wfar = hfar * aspect_ratio

# Create near plane points
x_near = [-wnear / 2, wnear / 2, wnear / 2, -wnear / 2, -wnear / 2]
y_near = [-hnear / 2, -hnear / 2, hnear / 2, hnear / 2, -hnear / 2]
z_near = [near] * 5

# Create far plane points  
x_far = [-wfar / 2, wfar / 2, wfar / 2, -wfar / 2, -wfar / 2]
y_far = [-hfar / 2, -hfar / 2, hfar / 2, hfar / 2, -hfar / 2]
z_far = [far] * 5

# Plot frustum
ax.plot(x_near, y_near, z_near, color='blue')
ax.plot(x_far, y_far, z_far, color='blue')
ax.plot([x_near[0], x_far[0]], [y_near[0], y_far[0]], [z_near[0], z_far[0]], color='blue')
ax.plot([x_near[1], x_far[1]], [y_near[1], y_far[1]], [z_near[1], z_far[1]], color='blue')  
ax.plot([x_near[2], x_far[2]], [y_near[2], y_far[2]], [z_near[2], z_far[2]], color='blue')
ax.plot([x_near[3], x_far[3]], [y_near[3], y_far[3]], [z_near[3], z_far[3]], color='blue')


# ax.scatter3D(mx, my, mz, color = "red")
ax.scatter3D(mx, my, mz, color = "green")

ax.scatter3D(bx, by, bz, color = "red")

ax.set_xlim(-6 , 6)
ax.set_ylim(-6 , 6)
ax.set_zlim(-10 , 10)

# Show plot
plt.show()
# %%
camera.world_to_ndc(np.array([1,0,near]))
for x, y, z in zip(mx, my, mz):
    print(camera.world_to_ndc(np.array([x,y,z])))
# %%


# %%
camera.world_to_ndc(np.array([0,0,far]))
# %%
camera.world_to_ndc(np.array([1,0,near]))
for x, y, z in zip(mx, my, mz):
    if(z < near):
        print(camera.world_to_ndc(np.array([x,y,z])))
    # print(camera.world_to_ndc(np.array([x,y,z])))
# %%

cx = []
cy = []
cz = []
count = 0
for gaussian in tqdm(sorted_gaussians):
    cx.append(gaussian.camera_space_pos[0])
    cy.append(gaussian.camera_space_pos[1])
    cz.append(gaussian.camera_space_pos[2])
    count += 1
    if count > 5000:
        break
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

# Creating plot
# ax.scatter3D(mx, my, mz, color = "green")
ax.scatter3D(cx, cy, cz, color = "red")

ax.set_xlim(-3 , 3)
ax.set_ylim(-3 , 3)
#ax.set_zlim(0 , 5)
plt.title("simple 3D scatter plot")
plt.show()
# %%
print(gaussians.gaussians[0].camera_space_pos)
# %%
camera.is_inside_frustum(gaussians.gaussians[0].position)
# %%
%matplotlib widget
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Set up figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set camera parameters
aspect_ratio = 1

# Calculate width and height of near and far plane 
hnear = 2 * np.tan(np.deg2rad(fov / 2)) * near
wnear = hnear * aspect_ratio
hfar = 2 * np.tan(np.deg2rad(fov / 2)) * far
wfar = hfar * aspect_ratio

# Create near plane points
x_near = [-wnear / 2, wnear / 2, wnear / 2, -wnear / 2, -wnear / 2]
y_near = [-hnear / 2, -hnear / 2, hnear / 2, hnear / 2, -hnear / 2]
z_near = [near] * 5

# Create far plane points  
x_far = [-wfar / 2, wfar / 2, wfar / 2, -wfar / 2, -wfar / 2]
y_far = [-hfar / 2, -hfar / 2, hfar / 2, hfar / 2, -hfar / 2]
z_far = [far] * 5

# Plot frustum
ax.plot(x_near, y_near, z_near, color='blue')
ax.plot(x_far, y_far, z_far, color='blue')
ax.plot([x_near[0], x_far[0]], [y_near[0], y_far[0]], [z_near[0], z_far[0]], color='blue')
ax.plot([x_near[1], x_far[1]], [y_near[1], y_far[1]], [z_near[1], z_far[1]], color='blue')  
ax.plot([x_near[2], x_far[2]], [y_near[2], y_far[2]], [z_near[2], z_far[2]], color='blue')
ax.plot([x_near[3], x_far[3]], [y_near[3], y_far[3]], [z_near[3], z_far[3]], color='blue')


view_space = camera.world_to_view(gaussians.gaussians[1].position)
# ax.scatter3D(mx, my, mz, color = "red")
ax.scatter3D([view_space[0]], [view_space[1]], [view_space[2]], color = "green")

# ax.scatter3D(bx, by, bz, color = "red")


# Show plot
plt.show()
# %%
pos = gaussians.gaussians[1].position
view_pos = camera.world_to_view(pos)

print(view_pos)
print(camera.projection_matrix @ np.concatenate((view_pos, np.array([1])), axis=0))
print(near <= view_pos[2] <= far)
# pos[2] = -pos[2]
print(camera.world_to_ndc(pos))
print(camera.is_inside_frustum(pos))
# %%
print(camera.world_to_camera_matrix)
# %%
print(camera.projection_matrix)
x = np.array([0.5,0.5, 1,1])
print(camera.projection_matrix.shape)
print(x.shape)
y = np.linalg.inv(camera.view_matrix) @ x
print("y", y)
ndc = (camera.projection_matrix @ camera.view_matrix @ y)
print(ndc / ndc[3])
y = y[:3] / y[3]
print(camera.world_to_ndc(y))
print(camera.is_inside_frustum(y))
# %%
