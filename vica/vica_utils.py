import torch
from torchtyping import TensorType
import nerfstudio.utils.poses as pose_utils
from enum import Enum, auto
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CameraType



def radial_and_tangential_distort(coords, distortion_params):
    """
    Applies radial and tangential distortion to 2D coordinates.

    Args:
        coords: The undistorted coordinates (a torch.Tensor).
        distortion_params: The distortion parameters [k1, k2, k3, k4, p1, p2].

    Returns:
        The distorted coordinates (a torch.Tensor).
    """
    
    k1, k2, k3, k4, p1, p2 = distortion_params.squeeze()

    x = coords[..., 0]
    y = coords[..., 1]

    r2 = x**2 + y**2
    radial_distortion = k1 * r2 + k2 * r2**2 + k3 * r2**3 + k4 * r2**4

    x_distorted = x * (1 + radial_distortion) + 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
    y_distorted = y * (1 + radial_distortion) + p1 * (r2 + 2 * y**2) + 2 * p2 * x * y

    return torch.stack([x_distorted, y_distorted], dim=-1)

def generate_coords_from_directions(
    ref_camera,
    camera_indices: TensorType["num_rays":..., "num_cameras_batch_dims"],

    directions,
    camera_opt_to_camera
    ):
    # print(ref_camera.camera_type, ref_camera.distortion_params)
    true_indices = [camera_indices[..., i] for i in range(camera_indices.shape[-1])]
    num_rays_shape = camera_indices.shape[:-1]
    cam_types = torch.unique(ref_camera.camera_type, sorted=False)
    
    # print(self.camera_to_worlds.shape)
    c2w = ref_camera.camera_to_worlds.unsqueeze(0)
    # print(c2w.shape)
    assert c2w.shape == num_rays_shape + (3, 4)

    if camera_opt_to_camera is not None:
        c2w = pose_utils.multiply(c2w, camera_opt_to_camera)
    rotation = c2w[..., :3, :3]  # (..., 3, 3)

    assert rotation.shape == num_rays_shape + (3, 3)
    
    directions = torch.matmul(directions, rotation)

    if CameraType.PERSPECTIVE.value in cam_types:
        directions = directions/(directions[...,2]/(-1))[...,None]
        coords  = torch.zeros(*directions.shape[:-1],2, device=directions.device)
        coords[..., 0] =  directions[...,0]
        coords[..., 1] = directions[...,1]     
    else:
        # print(CameraType.EQUIRECTANGULAR.value)
        raise NotImplementedError(f"Camera should be set as PERSPECTIVE, not{cam_types}")
    
    fx, fy = ref_camera.fx[true_indices].squeeze(-1), ref_camera.fy[true_indices].squeeze(-1)  # (num_rays,)
    cx, cy = ref_camera.cx[true_indices].squeeze(-1), ref_camera.cy[true_indices].squeeze(-1)  # (num_rays,)

    new_coords = torch.zeros_like(coords)
    new_coords[..., 0] = coords[..., 0]*fx + cx
    new_coords[..., 1] = -coords[..., 1]*fy + cy

    return new_coords