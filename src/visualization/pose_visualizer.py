"""
Camera Pose Visualizer using Open3D

This module provides visualization utilities for camera poses and trajectories
in 3D space using Open3D, based on the implementation from capture_and_match.ipynb.
"""

import numpy as np
import open3d as o3d
from typing import List, Dict, Any, Optional, Tuple


class PoseVisualizer:
    """
    Camera pose visualizer for SLAM systems using Open3D.
    
    This class provides methods to visualize camera poses, trajectories,
    and related geometric elements in 3D space.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize pose visualizer.
        
        Args:
            verbose: Enable verbose output (default: False)
        """
        self.verbose = verbose
        self.geometries = []
        
        if self.verbose:
            print("PoseVisualizer initialized")
    
    def clear_geometries(self):
        """Clear all stored geometries."""
        self.geometries = []
        if self.verbose:
            print("Cleared all geometries")
    
    def create_camera_visualization(self, R: np.ndarray, t: np.ndarray, scale: float = 0.05) -> List[o3d.geometry.TriangleMesh]:
        """
        Create Open3D geometric objects for camera pose visualization.
        
        Based on the notebook implementation with absolute scale camera positioning.
        
        Args:
            R: Rotation matrix (3x3)
            t: Translation vector (3x1)
            scale: Visualization scale factor (default: 0.05)
            
        Returns:
            List of Open3D geometric objects
        """
        geometries = []
        
        # Calculate absolute camera position
        camera_pos = -R.T @ t.reshape(3, 1)
        camera_pos = camera_pos.flatten()
        
        if self.verbose:
            print(f"Camera position: [{camera_pos[0]:.3f}, {camera_pos[1]:.3f}, {camera_pos[2]:.3f}]m")
        
        # Camera position as sphere
        camera_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=scale * 1.5)
        camera_sphere.translate(camera_pos)
        camera_sphere.paint_uniform_color([0, 1, 0])  # Green
        geometries.append(camera_sphere)
        
        # Camera coordinate frame
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale * 2)
        # Apply camera pose transformation
        transform = np.eye(4)
        transform[:3, :3] = R.T
        transform[:3, 3] = camera_pos
        camera_frame.transform(transform)
        geometries.append(camera_frame)
        
        return geometries
    
    def create_movement_arrow(self, start_pos: np.ndarray, end_pos: np.ndarray, scale: float = 0.05) -> List[o3d.geometry.TriangleMesh]:
        """
        Create arrow visualization for camera movement.
        
        Args:
            start_pos: Starting position (3,)
            end_pos: Ending position (3,)
            scale: Visualization scale factor
            
        Returns:
            List of Open3D geometric objects for the arrow
        """
        geometries = []
        
        # Movement path line
        movement_points = np.array([start_pos, end_pos])
        movement_lines = np.array([[0, 1]])
        movement_line_set = o3d.geometry.LineSet()
        movement_line_set.points = o3d.utility.Vector3dVector(movement_points)
        movement_line_set.lines = o3d.utility.Vector2iVector(movement_lines)
        movement_line_set.paint_uniform_color([1, 0.5, 0])  # Orange
        geometries.append(movement_line_set)
        
        # Arrow head (cone)
        movement_vector = end_pos - start_pos
        if np.linalg.norm(movement_vector) > 1e-6:
            arrow_direction = movement_vector / np.linalg.norm(movement_vector)
            arrow_head_pos = end_pos - arrow_direction * scale * 0.5
            
            arrow_head = o3d.geometry.TriangleMesh.create_cone(radius=scale * 2, height=scale * 3)
            
            # Rotate arrow to point in movement direction
            z_axis = np.array([0, 0, 1])
            rotation_axis = np.cross(z_axis, arrow_direction)
            if np.linalg.norm(rotation_axis) > 1e-6:
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                rotation_angle = np.arccos(np.clip(np.dot(z_axis, arrow_direction), -1, 1))
                rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
                arrow_head.rotate(rotation_matrix, center=(0, 0, 0))
            
            arrow_head.translate(arrow_head_pos)
            arrow_head.paint_uniform_color([1, 0.5, 0])  # Orange
            geometries.append(arrow_head)
        
        return geometries
    
    def create_projection_rays(self, camera_center: np.ndarray, points_3d: np.ndarray, 
                             color: List[float] = [0, 1, 1], max_rays: int = 8) -> List[o3d.geometry.LineSet]:
        """
        Create projection rays from camera to 3D points.
        
        Args:
            camera_center: Camera center position (3,)
            points_3d: 3D points array (N, 3)
            color: RGB color for rays (default: [0, 1, 1] - cyan)
            max_rays: Maximum number of rays to draw (default: 8)
            
        Returns:
            List of Open3D LineSet objects
        """
        geometries = []
        
        if len(points_3d) == 0:
            return geometries
        
        # Filter valid points
        valid_points = points_3d[np.all(np.isfinite(points_3d), axis=1)]
        if len(valid_points) == 0:
            return geometries
        
        # Select representative points
        distances = np.linalg.norm(valid_points - camera_center, axis=1)
        median_distance = np.median(distances)
        
        # Points near median distance
        distance_mask = np.abs(distances - median_distance) < median_distance * 0.5
        representative_points = valid_points[distance_mask]
        
        # Sample points
        sample_indices = np.linspace(0, len(representative_points)-1, 
                                   min(max_rays, len(representative_points)), dtype=int)
        
        for idx in sample_indices:
            if idx < len(representative_points):
                point_3d = representative_points[idx]
                
                # Create line from camera to 3D point
                line_points = np.array([camera_center, point_3d])
                line_indices = np.array([[0, 1]])
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(line_points)
                line_set.lines = o3d.utility.Vector2iVector(line_indices)
                line_set.paint_uniform_color(color)
                geometries.append(line_set)
        
        return geometries
    
    def create_measurement_annotations(self, t: np.ndarray, euler_angles: np.ndarray, 
                                     scale: float = 0.01) -> List[o3d.geometry.TriangleMesh]:
        """
        Create visual annotations for measurement results.
        
        Args:
            t: Translation vector (3x1)
            euler_angles: Euler angles array (3,)
            scale: Scale for annotation objects
            
        Returns:
            List of annotation geometries
        """
        geometries = []
        
        # Information sphere at movement midpoint
        mid_point = t.flatten() / 2
        info_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=scale)
        info_sphere.translate(mid_point)
        info_sphere.paint_uniform_color([1, 1, 0])  # Yellow
        geometries.append(info_sphere)
        
        return geometries
    
    def visualize_absolute_pose(self, R: np.ndarray, t: np.ndarray, points_3d: Optional[np.ndarray] = None,
                              origin_pcd: Optional[o3d.geometry.PointCloud] = None,
                              title: str = "Camera Pose Estimation Results",
                              window_size: Tuple[int, int] = (1200, 900)) -> Any:
        """
        Create complete visualization of absolute pose estimation results.
        
        Based on the comprehensive visualization from the notebook.
        
        Args:
            R: Rotation matrix (3x3)
            t: Translation vector (3x1)
            points_3d: Optional reconstructed 3D points (N, 3)
            origin_pcd: Optional original point cloud
            title: Visualization window title
            window_size: Window size (width, height)
            
        Returns:
            Plotly figure object
        """
        self.clear_geometries()
        
        # Adaptive visualization scale
        visualization_scale = max(0.02, np.linalg.norm(t) * 0.2)
        
        if self.verbose:
            print(f"Visualization scale: {visualization_scale:.4f}")
        
        # 1. Original point cloud (if provided)
        if origin_pcd is not None:
            if self.verbose:
                print(f"Processing original point cloud with {len(origin_pcd.points)} points")
            
            # Downsample if too large
            if len(origin_pcd.points) > 8000:
                pcd_downsampled = origin_pcd.voxel_down_sample(voxel_size=0.015)
                if self.verbose:
                    print(f"Downsampled: {len(origin_pcd.points)} -> {len(pcd_downsampled.points)}")
            else:
                pcd_downsampled = origin_pcd
            
            self.geometries.append(pcd_downsampled)
        
        # 2. Reconstructed 3D points (if provided)
        if points_3d is not None and len(points_3d) > 0:
            if self.verbose:
                print("Processing reconstructed 3D points")
            
            reconstructed_pcd = self._create_3d_points_cloud(points_3d, color=[1, 0, 0])
            if len(reconstructed_pcd.points) > 0:
                self.geometries.append(reconstructed_pcd)
                
                # Add some points as spheres for better visibility
                points_array = np.asarray(reconstructed_pcd.points)
                for i, point in enumerate(points_array[::3]):  # Every 3rd point
                    if i < 20:  # Max 20 spheres
                        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=visualization_scale * 0.3)
                        sphere.translate(point)
                        sphere.paint_uniform_color([1, 0.2, 0.2])  # Light red
                        self.geometries.append(sphere)
        
        # 3. Origin camera (first camera)
        if self.verbose:
            print("Creating origin camera visualization")
        
        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=visualization_scale * 3)
        origin_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=visualization_scale * 1.2)
        origin_sphere.paint_uniform_color([1, 0, 0])  # Red
        self.geometries.extend([origin_frame, origin_sphere])
        
        # 4. Second camera visualization
        if self.verbose:
            print("Creating second camera visualization")
        
        camera_geometries = self.create_camera_visualization(R, t, scale=visualization_scale)
        self.geometries.extend(camera_geometries)
        
        # 5. Movement arrow
        origin_pos = np.array([0, 0, 0])
        camera_pos = (-R.T @ t.flatten()).flatten()
        arrow_geometries = self.create_movement_arrow(origin_pos, camera_pos, scale=visualization_scale)
        self.geometries.extend(arrow_geometries)
        
        # 6. Projection rays (if 3D points available)
        if points_3d is not None and len(points_3d) > 0:
            if self.verbose:
                print("Creating projection rays")
            
            try:
                # Rays from origin camera
                rays1 = self.create_projection_rays(origin_pos, points_3d, color=[0, 1, 1])
                self.geometries.extend(rays1)
                
                # Rays from second camera
                rays2 = self.create_projection_rays(camera_pos, points_3d, color=[1, 1, 0])
                self.geometries.extend(rays2)
                
            except Exception as e:
                if self.verbose:
                    print(f"Error creating projection rays: {e}")
        
        # 7. Measurement annotations
        euler_angles = self._rotation_matrix_to_euler_angles(R)
        annotation_geometries = self.create_measurement_annotations(t, euler_angles)
        self.geometries.extend(annotation_geometries)
        
        if self.verbose:
            print(f"Created {len(self.geometries)} visualization objects")
        
        # Create Open3D Plotly visualization
        fig = o3d.visualization.draw_plotly(
            self.geometries,
            window_name=title,
            width=window_size[0],
            height=window_size[1]
        )
        
        return fig
    
    def _create_3d_points_cloud(self, points_3d: np.ndarray, color: List[float] = [1, 0, 0]) -> o3d.geometry.PointCloud:
        """
        Create point cloud from 3D points with filtering.
        
        Args:
            points_3d: 3D points array (N, 3)
            color: RGB color for points
            
        Returns:
            Open3D PointCloud object
        """
        if len(points_3d) == 0:
            return o3d.geometry.PointCloud()
        
        # Filter valid points
        finite_mask = np.all(np.isfinite(points_3d), axis=1)
        valid_points = points_3d[finite_mask]
        
        if len(valid_points) == 0:
            return o3d.geometry.PointCloud()
        
        # Remove outliers
        distances = np.linalg.norm(valid_points, axis=1)
        q75 = np.percentile(distances, 75)
        q25 = np.percentile(distances, 25)
        iqr = q75 - q25
        upper_bound = q75 + 1.5 * iqr
        
        # Keep points within reasonable range (max 5m)
        reasonable_mask = distances < min(upper_bound, 5.0)
        filtered_points = valid_points[reasonable_mask]
        
        if self.verbose:
            print(f"3D points: {len(points_3d)} -> {len(filtered_points)} (after filtering)")
        
        if len(filtered_points) == 0:
            return o3d.geometry.PointCloud()
        
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
        point_cloud.paint_uniform_color(color)
        return point_cloud
    
    def _rotation_matrix_to_euler_angles(self, R: np.ndarray) -> np.ndarray:
        """Simple rotation matrix to Euler angles conversion for internal use."""
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        return np.array([x, y, z]) * 180.0 / np.pi
    
    def print_legend(self, t: np.ndarray, scale_factor: float = 1.0):
        """
        Print visualization legend and measurement summary.
        
        Args:
            t: Translation vector
            scale_factor: Scale factor used in pose estimation
        """
        print("\n=== Visualization Legend ===")
        print("Blue points: RealSense actual scene point cloud")
        print("Red points/spheres: Reconstructed 3D feature points")
        print("Red sphere + RGB axes: First camera (origin)")
        print("Green sphere + RGB axes: Second camera (absolute position)")
        print("Orange arrow: Camera movement path (actual distance)")
        print("Yellow sphere: Movement path midpoint")
        print("Cyan lines: First camera projection rays")
        print("Yellow lines: Second camera projection rays")
        
        print(f"\n=== Measurement Summary ===")
        movement_distance = np.linalg.norm(t)
        print(f"Actual movement distance: {movement_distance:.4f}m ({movement_distance*100:.1f}cm)")
        t_flat = t.flatten()
        print(f"Movement direction: [{t_flat[0]:.3f}, {t_flat[1]:.3f}, {t_flat[2]:.3f}]")
        print(f"Scale factor: {scale_factor:.4f}")


def create_pose_visualizer(verbose: bool = False) -> PoseVisualizer:
    """
    Factory function to create a pose visualizer.
    
    Args:
        verbose: Enable verbose output
        
    Returns:
        PoseVisualizer instance
    """
    return PoseVisualizer(verbose=verbose)