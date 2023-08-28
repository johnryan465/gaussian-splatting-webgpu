import numpy as np
from reference.camera import Camera

from reference.loadfile import Gaussian



SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = np.array([
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
])

SH_C3 = np.array([
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
])
def computeColorFromSH(deg: int,  pos: np.ndarray[3], campos: np.ndarray[3], sh: np.ndarray) -> np.ndarray[3]:
    # The implementation is loosely based on code for
    # "Differentiable Point-Based Radiance Fields for
    # Efficient View Synthesis" by Zhang et al. (2022)
    dir = pos - campos
    dir = dir / np.linalg.norm(dir)

    result = SH_C0 * sh[0]

    if (deg > 0):
        x = dir[0]
        y = dir[1]
        z = dir[2]
        result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3]

    if (deg > 1):
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        yz = y * z
        xz = x * z
        result = result + SH_C2[0] * xy * sh[4] + SH_C2[1] * yz * sh[5] + SH_C2[2] * (2.0 * zz - xx - yy) * sh[6] + SH_C2[3] * xz * sh[7] + SH_C2[4] * (xx - yy) * sh[8]

    if (deg > 2):
        result = result + SH_C3[0] * y * (3.0 * xx - yy) * sh[9] + SH_C3[1] * xy * z * sh[10] + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh[11] + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[12] + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh[13] + SH_C3[5] * z * (xx - yy) * sh[14] + SH_C3[6] * x * (xx - 3.0 * yy) * sh[15]

    result += 0.5

    return np.clip(result, a_min=0.0, a_max=None)





