
from dataclasses import dataclass
import numpy as np



def rotation_translation(rotation: np.ndarray[3, 3], translation: np.ndarray[3]) -> np.ndarray[4, 4]:
    """Return a rotation translation matrix"""
    return np.array(
        [  [rotation[0][0], rotation[0][1], rotation[0][2], translation[0]],
            [rotation[1][0], rotation[1][1], rotation[1][2], translation[1]],
            [rotation[2][0], rotation[2][1], rotation[2][2], translation[2]],
            [0, 0, 0, 1]]
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
        p = self.world_to_camera_matrix @ np.concatenate((point, np.array([1])), axis=0)
        return p[:3] / p[3]
    
    def world_to_ndc(self, point: np.ndarray[3]) -> np.ndarray[3]:
        """Convert a point in world space to ndc space"""
        p = self.world_to_screen @ np.concatenate((point, np.array([1])), axis=0)
        p = p[:3] / p[3]
        return p


    def is_inside_frustum(self, world_space: np.ndarray) -> bool:
        """Check if a point is inside a view frustum"""
        # ndc = self.world_to_ndc(world_space)
        # cam_space = self.world_to_view(world_space)
        p = self.world_to_screen @ np.concatenate((world_space, np.array([1])), axis=0)
        if p[3] < 0:
            return False
        ndc = p[:3] / p[3]
        return -1 <= ndc[0] <= 1 and -1 <= ndc[1] <= 1 and -1 <= ndc[2] <= 1

        
    
    
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
            [0, 0,  -(self.far + self.near) / (self.far - self.near), 2* self.far * self.near / (self.far - self.near)],
            [0, 0, 1, 0]])
    
    @property
    def world_to_camera_matrix(self) -> np.ndarray[4, 4]:
        """Return the world to camera matrix"""
        return self.view_matrix
    
    @property
    def camera_to_screen_matrix(self) -> np.ndarray[4, 4]:
        """Return the camera to screen matrix"""
        return self.projection_matrix
    
    @property
    def world_to_screen(self) -> np.ndarray[4, 4]:
        """Return the camera matrix"""
        return self.camera_to_screen_matrix @ self.world_to_camera_matrix