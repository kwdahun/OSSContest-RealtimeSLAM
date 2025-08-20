import numpy as np
from typing import Tuple, Optional


class CameraIntrinsics:
    def __init__(self, fx: float, fy: float, cx: float, cy: float, 
                 width: int, height: int):
        """
        Camera intrinsic parameters for Intel RealSense D435i.
        
        Args:
            fx, fy: Focal lengths in x and y
            cx, cy: Principal point coordinates
            width, height: Image dimensions
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.K_inv = np.linalg.inv(self.K)
    
    def pixel_to_camera(self, u: float, v: float, depth: float) -> np.ndarray:
        """
        Convert pixel coordinates to 3D camera coordinates.
        
        Args:
            u, v: Pixel coordinates
            depth: Depth value in meters
            
        Returns:
            3D point in camera coordinate system
        """
        if depth <= 0:
            return None
            
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth
        
        return np.array([x, y, z], dtype=np.float32)
    
    def camera_to_pixel(self, point_3d: np.ndarray) -> Tuple[int, int]:
        """
        Project 3D camera coordinates to pixel coordinates.
        
        Args:
            point_3d: 3D point in camera coordinate system
            
        Returns:
            Pixel coordinates (u, v)
        """
        x, y, z = point_3d
        
        if z <= 0:
            return None
            
        u = int(self.fx * x / z + self.cx)
        v = int(self.fy * y / z + self.cy)
        
        return u, v
    
    def is_valid_pixel(self, u: int, v: int) -> bool:
        """Check if pixel coordinates are within image bounds."""
        return 0 <= u < self.width and 0 <= v < self.height
    
    def get_camera_matrix(self) -> np.ndarray:
        """Get the camera matrix K."""
        return self.K


def create_default_d435i_intrinsics() -> CameraIntrinsics:
    """Create default intrinsics for Intel RealSense D435i (approximate values)."""
    return CameraIntrinsics(
        fx=615.0, fy=615.0,
        cx=320.0, cy=240.0,
        width=640, height=480
    )