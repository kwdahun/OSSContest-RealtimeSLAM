"""
Point Cloud Visualizer using Open3D

This module provides visualization utilities for point clouds from RealSense cameras
and reconstructed 3D points, based on the implementation from capture_and_match.ipynb.
"""

import numpy as np
import cv2
import open3d as o3d
from typing import Dict, Any, Optional, Tuple, List


class PointCloudVisualizer:
    """
    Point cloud visualizer for SLAM systems using Open3D.
    
    This class provides methods to create and visualize point clouds from
    RGB-D data and reconstructed 3D points.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize point cloud visualizer.
        
        Args:
            verbose: Enable verbose output (default: False)
        """
        self.verbose = verbose
        
        if self.verbose:
            print("PointCloudVisualizer initialized")
    
    def create_point_cloud_from_rgbd(self, color_image: np.ndarray, depth_image: np.ndarray,
                                   camera_intrinsics: Dict[str, float], depth_scale: float = 0.001,
                                   depth_trunc: float = 5.0) -> o3d.geometry.PointCloud:
        """
        Create Open3D point cloud from RGB-D images.
        
        Based on the notebook implementation using RealSense camera data.
        
        Args:
            color_image: RGB color image (H, W, 3)
            depth_image: Depth image (H, W) already scaled to meters
            camera_intrinsics: Dictionary with fx, fy, cx, cy values
            depth_scale: Original depth scale factor (needed to convert back for Open3D)
            depth_trunc: Maximum depth to include in meters (default: 5.0)
            
        Returns:
            Open3D PointCloud object
        """
        height, width = color_image.shape[:2]
        
        # Create Open3D camera intrinsic object
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=width,
            height=height,
            fx=camera_intrinsics['fx'],
            fy=camera_intrinsics['fy'],
            cx=camera_intrinsics['cx'],
            cy=camera_intrinsics['cy']
        )
        
        # Convert images to Open3D format
        if color_image.shape[2] == 3 and color_image.dtype == np.uint8:
            # Assume BGR format from OpenCV, convert to RGB
            rgb_o3d = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        else:
            rgb_o3d = o3d.geometry.Image(color_image)
        
        # Convert depth back to raw sensor values for Open3D
        # Since depth_image is already scaled to meters, convert back to raw values
        raw_depth_image = (depth_image / depth_scale).astype(np.uint16)
        depth_o3d = o3d.geometry.Image(raw_depth_image)
        
        # Create RGBD image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d,
            depth_o3d,
            depth_scale=1.0 / depth_scale,  # Open3D expects inverse of scale
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False
        )
        
        # Create point cloud from RGBD image
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        
        # Transform point cloud (flip for better visualization)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        
        if self.verbose:
            print(f"Point cloud created with {len(pcd.points)} points")
            bounds = pcd.get_axis_aligned_bounding_box()
            print(f"Point cloud bounds: {bounds}")
        
        return pcd
    
    def downsample_point_cloud(self, pcd: o3d.geometry.PointCloud, voxel_size: float = 0.015,
                             max_points: int = 8000) -> o3d.geometry.PointCloud:
        """
        Downsample point cloud for better visualization performance.
        
        Args:
            pcd: Input point cloud
            voxel_size: Voxel size for downsampling (default: 0.015)
            max_points: Maximum points threshold for downsampling (default: 8000)
            
        Returns:
            Downsampled point cloud
        """
        if len(pcd.points) > max_points:
            pcd_downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
            if self.verbose:
                print(f"Downsampled: {len(pcd.points)} -> {len(pcd_downsampled.points)} points")
            return pcd_downsampled
        else:
            if self.verbose:
                print(f"No downsampling needed: {len(pcd.points)} points")
            return pcd
    
    def create_reconstructed_point_cloud(self, points_3d: np.ndarray, 
                                       color: List[float] = [1, 0, 0]) -> o3d.geometry.PointCloud:
        """
        Create point cloud from reconstructed 3D points with filtering.
        
        Args:
            points_3d: Reconstructed 3D points array (N, 3)
            color: RGB color for points (default: [1, 0, 0] - red)
            
        Returns:
            Open3D PointCloud object
        """
        if len(points_3d) == 0:
            return o3d.geometry.PointCloud()
        
        # Filter valid points (remove NaN and infinite values)
        finite_mask = np.all(np.isfinite(points_3d), axis=1)
        valid_points = points_3d[finite_mask]
        
        if len(valid_points) == 0:
            if self.verbose:
                print("No valid finite points found")
            return o3d.geometry.PointCloud()
        
        # Remove outliers using IQR method
        distances = np.linalg.norm(valid_points, axis=1)
        q75 = np.percentile(distances, 75)
        q25 = np.percentile(distances, 25)
        iqr = q75 - q25
        upper_bound = q75 + 1.5 * iqr
        
        # Keep points within reasonable range (max 5m)
        reasonable_mask = distances < min(upper_bound, 5.0)
        filtered_points = valid_points[reasonable_mask]
        
        if self.verbose:
            print(f"Reconstructed 3D points: {len(points_3d)} -> {len(filtered_points)} (after filtering)")
        
        if len(filtered_points) == 0:
            return o3d.geometry.PointCloud()
        
        # Create point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
        point_cloud.paint_uniform_color(color)
        
        return point_cloud
    
    def visualize_depth_images(self, depth_image1: np.ndarray, depth_image2: np.ndarray,
                             titles: Tuple[str, str] = ("Depth Image 1", "Depth Image 2"),
                             depth_range: Tuple[float, float] = (0.0, 5.0)) -> Any:
        """
        Visualize depth images that are already in meters.
        
        Args:
            depth_image1: First depth image (already in meters)
            depth_image2: Second depth image (already in meters)
            titles: Titles for the images
            depth_range: Depth range for visualization (min, max) in meters
            
        Returns:
            Dictionary with depth images and statistics
        """
        # Depth images are already in meters, no scaling needed
        
        if self.verbose:
            print(f"Depth images are already in meters")
            print(f"Depth 1 range: {depth_image1.min():.3f}m - {depth_image1.max():.3f}m")
            print(f"Depth 2 range: {depth_image2.min():.3f}m - {depth_image2.max():.3f}m")
        
        # Return the depth images (already in meters)
        return {
            'depth_image1': depth_image1,
            'depth_image2': depth_image2,
            'depth_stats': {
                'min_depth': depth_image1.min(),
                'max_depth': depth_image1.max(),
                'mean_depth': depth_image1.mean()
            }
        }
    
    def create_feature_point_spheres(self, points_3d: np.ndarray, scale: float = 0.01,
                                   color: List[float] = [1, 0.2, 0.2], 
                                   max_spheres: int = 20, skip_factor: int = 3) -> List[o3d.geometry.TriangleMesh]:
        """
        Create sphere representations of 3D feature points.
        
        Args:
            points_3d: 3D points array (N, 3)
            scale: Sphere radius scale
            color: RGB color for spheres
            max_spheres: Maximum number of spheres to create
            skip_factor: Skip every N points for performance
            
        Returns:
            List of Open3D sphere meshes
        """
        spheres = []
        
        if len(points_3d) == 0:
            return spheres
        
        # Filter valid points
        finite_mask = np.all(np.isfinite(points_3d), axis=1)
        valid_points = points_3d[finite_mask]
        
        # Sample points
        for i, point in enumerate(valid_points[::skip_factor]):
            if i >= max_spheres:
                break
                
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=scale)
            sphere.translate(point)
            sphere.paint_uniform_color(color)
            spheres.append(sphere)
        
        if self.verbose:
            print(f"Created {len(spheres)} feature point spheres")
        
        return spheres
    
    def combine_point_clouds(self, *point_clouds: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Combine multiple point clouds into one.
        
        Args:
            *point_clouds: Variable number of point clouds to combine
            
        Returns:
            Combined point cloud
        """
        if len(point_clouds) == 0:
            return o3d.geometry.PointCloud()
        
        combined_pcd = o3d.geometry.PointCloud()
        
        for pcd in point_clouds:
            if len(pcd.points) > 0:
                combined_pcd += pcd
        
        if self.verbose:
            print(f"Combined {len(point_clouds)} point clouds into one with {len(combined_pcd.points)} points")
        
        return combined_pcd
    
    def apply_statistical_outlier_removal(self, pcd: o3d.geometry.PointCloud,
                                        nb_neighbors: int = 20, std_ratio: float = 2.0) -> o3d.geometry.PointCloud:
        """
        Apply statistical outlier removal to point cloud.
        
        Args:
            pcd: Input point cloud
            nb_neighbors: Number of neighbors for analysis
            std_ratio: Standard deviation ratio threshold
            
        Returns:
            Filtered point cloud
        """
        if len(pcd.points) == 0:
            return pcd
        
        pcd_filtered, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        
        if self.verbose:
            print(f"Statistical outlier removal: {len(pcd.points)} -> {len(pcd_filtered.points)} points")
        
        return pcd_filtered
    
    def estimate_normals(self, pcd: o3d.geometry.PointCloud, 
                        search_param: o3d.geometry.KDTreeSearchParamHybrid = None) -> o3d.geometry.PointCloud:
        """
        Estimate normals for point cloud.
        
        Args:
            pcd: Input point cloud
            search_param: Search parameters for normal estimation
            
        Returns:
            Point cloud with normals
        """
        if search_param is None:
            search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        
        pcd.estimate_normals(search_param=search_param)
        
        # Orient normals consistently
        pcd.orient_normals_consistent_tangent_plane(k=10)
        
        if self.verbose:
            print(f"Estimated normals for {len(pcd.points)} points")
        
        return pcd
    
    def create_mesh_from_point_cloud(self, pcd: o3d.geometry.PointCloud,
                                   method: str = 'poisson') -> o3d.geometry.TriangleMesh:
        """
        Create mesh from point cloud using various reconstruction methods.
        
        Args:
            pcd: Input point cloud with normals
            method: Reconstruction method ('poisson', 'ball_pivoting')
            
        Returns:
            Reconstructed triangle mesh
        """
        if len(pcd.points) == 0:
            return o3d.geometry.TriangleMesh()
        
        if method == 'poisson':
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        elif method == 'ball_pivoting':
            # Estimate radius for ball pivoting
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 1.5 * avg_dist
            
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector([radius, radius * 2])
            )
        else:
            raise ValueError(f"Unknown reconstruction method: {method}")
        
        if self.verbose:
            print(f"Created mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
        
        return mesh
    
    def save_point_cloud(self, pcd: o3d.geometry.PointCloud, filename: str) -> bool:
        """
        Save point cloud to file.
        
        Args:
            pcd: Point cloud to save
            filename: Output filename (supports .pcd, .ply, .xyz formats)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            success = o3d.io.write_point_cloud(filename, pcd)
            if self.verbose:
                print(f"Point cloud {'saved' if success else 'failed to save'} to {filename}")
            return success
        except Exception as e:
            if self.verbose:
                print(f"Error saving point cloud: {e}")
            return False
    
    def load_point_cloud(self, filename: str) -> o3d.geometry.PointCloud:
        """
        Load point cloud from file.
        
        Args:
            filename: Input filename
            
        Returns:
            Loaded point cloud
        """
        try:
            pcd = o3d.io.read_point_cloud(filename)
            if self.verbose:
                print(f"Loaded point cloud with {len(pcd.points)} points from {filename}")
            return pcd
        except Exception as e:
            if self.verbose:
                print(f"Error loading point cloud: {e}")
            return o3d.geometry.PointCloud()


def create_point_cloud_visualizer(verbose: bool = False) -> PointCloudVisualizer:
    """
    Factory function to create a point cloud visualizer.
    
    Args:
        verbose: Enable verbose output
        
    Returns:
        PointCloudVisualizer instance
    """
    return PointCloudVisualizer(verbose=verbose)