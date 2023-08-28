import numpy as np
from reference.camera import Camera
from reference.loadfile import Gaussian


#// Forward version of 2D covariance matrix computation
def computeMeanCov2D(mean: np.ndarray[3], focal_x: float, focal_y: float, tan_fovx: float, tan_fovy: float, cov3D: np.ndarray[6], viewmatrix: np.ndarray[3,3]):
    # The following models the steps outlined by equations 29
    # and 31 in "EWA Splatting" (Zwicker et al., 2002).
    # Additionally considers aspect / scaling of viewport.
    # Transposes used to account for row-/column-major conventions.
    t = viewmatrix[:3,:3] @ mean
    # t = t / t[3]
    #print("T", t)
    new_mean = t[:2] / t[2]
    #print("mean", new_mean)

    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    txtz = t[0] / t[2]
    tytz = t[1] / t[2]
    t[0] = np.clip(txtz, a_min=-limx, a_max=limx) * t[2]
    t[1] = np.clip(tytz, a_min=-limy, a_max=limy) * t[2]

    #print("t", t)

    # print(t[0])
    # print(focal_x)
    # print(focal_y)


    J = np.array([
        [focal_x / t[2], 0.0, -(focal_x * t[0]) / (t[2] * t[2])],
        [0.0, focal_y / t[2], -(focal_y * t[1]) / (t[2] * t[2])]])
    # print("J", J)

    W = viewmatrix[:3,:3]
    # print("W", W)


    M = J @ W
    # print("M", M)

    cov = M @ cov3D @ M.T
    # print("cov3d", cov3D)
    # print("cov", cov)

    # Apply low-pass filter: every Gaussian should be at least
    # one pixel wide/high. Discard 3rd row and column.
    #cov[0][0] += 0.3
    #cov[1][1] += 0.3
    #print(mean)
    #print(cov)
    return new_mean, cov[:2,:2]

def computeCov3D(scale: np.ndarray[3], mod: float, rot: np.ndarray[4]) -> np.ndarray[3,3]:
    # Create scaling matrix
    S = np.eye(3)
    S[0][0] = mod * scale[0]
    S[1][1] = mod * scale[1]
    S[2][2] = mod * scale[2]

    # Normalize quaternion to get valid rotation
    q = rot / np.linalg.norm(rot) # / glm::length(rot)
    r = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    # Compute rotation matrix from quaternion
    R = np.array([
        [1. - 2. * (y * y + z * z), 2. * (x * y - r * z), 2. * (x * z + r * y)],
        [2. * (x * y + r * z), 1. - 2. * (x * x + z * z), 2. * (y * z - r * x)],
        [2. * (x * z - r * y), 2. * (y * z + r * x), 1. - 2. * (x * x + y * y)]])

    M = S @ R

    # Compute 3D world covariance matrix Sigma
    Sigma = M.T @ M

    #print(Sigma)

    return Sigma[:3,:3]


def compute_exp_precompute(gaussian: Gaussian, camera: Camera):
    conv3d = computeCov3D(
        gaussian.scale,
        1.0,
        gaussian.rotQuat,
    )

    mean2d, conv2d = computeMeanCov2D(
        gaussian.position,
        camera.fovx,
        camera.fovy,
        np.tan(camera.fovx / 2),
        np.tan(camera.fovy / 2),
        conv3d,
        viewmatrix=camera.camera_matrix,
    )
    return mean2d, conv2d

def compute_exp_factor(gaussian: Gaussian, camera: Camera, x: float, y: float):
    conv3d = computeCov3D(
        gaussian.scale,
        1.0,
        gaussian.rotQuat,
    )

    mean2d, conv2d = computeMeanCov2D(
        gaussian.position,
        camera.fovx,
        camera.fovy,
        np.tan(camera.fovx / 2),
        np.tan(camera.fovy / 2),
        conv3d,
        viewmatrix=camera.camera_matrix,
    )

    distance_from_mean = np.array([x, y]) - mean2d
    exp_term = np.exp(-0.5 *  distance_from_mean.T @ np.linalg.inv(conv2d) @ distance_from_mean)
    return exp_term