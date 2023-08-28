
from dataclasses import dataclass
import numpy as np


# Convert to python
def getProjectionMatrix(znear: float, zfar: float, fovX: float, fovY: float) -> np.ndarray[4, 4]:
    """Return the projection matrix"""
    tanHalfFovY = np.tan(fovY / 2)
    tanHalfFovX = np.tan(fovX / 2)


    P = np.zeros((4, 4))

    z_sign = 1

    P[0, 0] = 1 / (tanHalfFovY)
    P[1, 1] =  1 / (tanHalfFovX)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def rotation_translation(rotation: np.ndarray[3, 3], translation: np.ndarray[3]) -> np.ndarray[4, 4]:
    """Return a rotation translation matrix"""
    return np.concatenate(
        (np.concatenate((rotation, translation.reshape(3, 1)), axis=1),
        np.array([[0, 0, 0, 1]])),
        axis=0
    )

@dataclass
class Camera:
    position: np.ndarray[3]
    rotation: np.ndarray[3, 3]
    fovx: float
    fovy: float
    near: float = 1
    far: float = 100.0
    width: int = 4
    height: int = 4

    
    @property
    def tanHalfFovX(self) -> float:
        """Return the tan of half fov in x"""
        return np.tan(self.fovx / 2)
    
    @property
    def tanHalfFovY(self) -> float:
        """Return the tan of half fov in y"""
        return np.tan(self.fovy / 2)

    
    def world_to_view(self, point: np.ndarray[3]) -> np.ndarray[3]:
        p = self.view_matrix @ np.concatenate((point, np.array([1])), axis=0)
        return p[:3] / p[3]
    
    def world_to_ndc(self, point: np.ndarray[3]) -> np.ndarray[3]:
        """Convert a point in world space to ndc space"""
        p = self.camera_matrix @ np.concatenate((point, np.array([1])), axis=0)
        return p[:3] / p[3]


    def is_inside_frustum(self, world_space: np.ndarray) -> bool:
        """Check if a point is inside a view frustum"""
        ndc = self.world_to_ndc(world_space)
        print(ndc)
        # print(ndc / ndc[2])
        # point = np.concatenate((point, np.array([0])), axis=0)
        if ndc[2] < 0:
            return False
        # if ndc[2] > 1:
        #    return False
        if ndc[0] > 1 or ndc[0] < -1:
            return False
        if ndc[1] > 1 or ndc[1] < -1:
            return False
        return True
        
    
    
    @property
    def view_matrix(self) -> np.ndarray:
        """Return the view matrix"""
        return rotation_translation(self.rotation, -self.position)
    
    @property
    def projection_matrix(self) -> np.ndarray[4, 4]:
        """Return the projection matrix"""
        return np.array([
            [ 1 / self.tanHalfFovX, 0, 0, 0],
            [0, 1/ self.tanHalfFovY,  0, 0],
            [0, 0,  -(self.far + self.near) / (self.far - self.near), - 2* self.far * self.near / (self.far - self.near)],
            [0, 0, -1, 0]]).T
    
    @property
    def camera_matrix(self) -> np.ndarray[4, 4]:
        """Return the camera matrix"""
        return self.projection_matrix @ self.view_matrix