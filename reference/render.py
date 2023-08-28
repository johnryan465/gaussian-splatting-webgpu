import numpy as np
from tqdm import tqdm
from reference.camera import Camera
from reference.loadfile import Gaussians
from reference.gaussians import compute_exp_precompute
from reference.utils import computeColorFromSH




def ndc2Pix(v: float,  S: int):
    return ((v + 1.0) * S - 1.0) * 0.5

class Renderer:
    def __init__(self) -> None:
        pass

    def render(self, width: int, height: int, camera: Camera, gaussians: Gaussians) -> np.ndarray:
        """Render a scene from a camera"""
        # Filter gaussian points that are outside the view frustum
        filtered_gaussians = []
        for gaussian in gaussians.gaussians:
            pos = gaussian.position
            if camera.is_inside_frustum(pos):
                # camera_space_pos_4d = camera.camera_matrix @ np.concatenate((pos, np.array([1])), axis=0)
                gaussian.camera_space_pos = camera.world_to_ndc(pos)
                filtered_gaussians.append(gaussian)

        # Sort gaussians by distance to camera
        sorted_gaussians = sorted(filtered_gaussians, key=lambda gaussian: -gaussian.camera_space_pos[2])

        # Render gaussians
        img = np.zeros((width, height, 3))
        alpha = np.ones((width, height))
        print(camera.world_to_screen)


        pixels = [(x, y) for x in range(width) for y in range(height)]
        count = 0
        for gaussian in tqdm(sorted_gaussians):
            colour = computeColorFromSH(
                3,
                gaussian.position,
                camera.position,
                gaussian.shCoeffs
            )
            colour = np.clip(colour, a_min=0.0, a_max=1.0)
            conv2d = compute_exp_precompute(gaussian, camera)
            mean2d = gaussian.camera_space_pos[:2]
            inv_cov2d = np.linalg.inv(conv2d)
            for (x,y) in pixels:
                if alpha[x, y] < 0.01:
                    continue
                camera_space_x =  (2 * (2 * x + 1) / (width * 2) - 1)
                camera_space_y =  (2 * (2 * y + 1) / (height * 2) - 1)
                # exp_term = compute_exp_factor(gaussian, camera, camera_space_x, camera_space_y)
                distance_from_mean = np.array([camera_space_x, camera_space_y]) - mean2d
                exp_term = np.exp(-0.5 *  distance_from_mean.T @ inv_cov2d @ distance_from_mean)
                curr_alpha = gaussian.opacity * exp_term
                if curr_alpha < 1/255:
                    continue
                curr_alpha = min(curr_alpha, 0.99)
                weighted_sum = curr_alpha * alpha[x, y] * colour
                alpha[x, y] *= (1 - curr_alpha)
                img[x,y] += weighted_sum
                if(alpha[x, y] < 0.01):
                   pixels.remove((x,y))
                # img[x,y] = pixel_value
            if len(pixels) == 0:
                break
            count += 1
            if count > 5000:
                break
        return img
                    
