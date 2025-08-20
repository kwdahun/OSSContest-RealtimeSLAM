"""
Frame Matcher for ED-SLAM System

This module provides a high-level API for matching RGB-depth frames,
estimating camera poses, and combining point clouds. Based on the 
implementation from modular_capture_and_match.ipynb.
"""

import numpy as np
import cv2
import open3d as o3d
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from features.orb_extractor import ORBExtractor
from features.orb_matcher import ORBMatcher, create_orb_matcher
from pose_estimation.absolute_pose import (
    estimate_absolute_pose_realsense,
    calculate_motion_metrics
)
from utils.rotation_utils import rotation_matrix_to_euler_angles
from visualization.pose_visualizer import PoseVisualizer, create_pose_visualizer
from visualization.point_cloud_visualizer import PointCloudVisualizer, create_point_cloud_visualizer


class FrameMatcher:
    """
    High-level API for matching two RGB-depth frames and combining point clouds.
    
    This class encapsulates the complete pipeline:
    1. Feature extraction using ORB
    2. Feature matching between frames
    3. Pose estimation with absolute scale
    4. Point cloud creation and combination
    5. Optional visualization
    """
    
    def __init__(self, camera_intrinsics: Dict[str, float], depth_scale: float = 0.001,
                 orb_features: int = 500, orb_scale_factor: float = 1.2, orb_levels: int = 8,
                 match_distance_threshold: float = 50.0, verbose: bool = False):
        """
        Initialize FrameMatcher with camera parameters and feature extraction settings.
        
        Args:
            camera_intrinsics: Dictionary with fx, fy, cx, cy values
            depth_scale: Depth scale factor from camera (default: 0.001 for RealSense)
            orb_features: Number of ORB features to extract (default: 500)
            orb_scale_factor: ORB scale factor (default: 1.2)
            orb_levels: Number of ORB pyramid levels (default: 8)
            match_distance_threshold: Feature matching distance threshold (default: 50.0)
            verbose: Enable verbose output (default: False)
        """
        self.camera_intrinsics = camera_intrinsics
        self.depth_scale = depth_scale
        self.verbose = verbose
        
        # Create camera intrinsic matrix
        self.K = np.array([
            [camera_intrinsics['fx'], 0, camera_intrinsics['cx']],
            [0, camera_intrinsics['fy'], camera_intrinsics['cy']],
            [0, 0, 1]
        ])
        
        # Initialize components
        self.orb_extractor = ORBExtractor(
            n_features=orb_features,
            scale_factor=orb_scale_factor,
            n_levels=orb_levels
        )
        
        self.orb_matcher = create_orb_matcher(
            distance_threshold=match_distance_threshold,
            cross_check=True,
            verbose=verbose
        )
        
        self.pose_visualizer = create_pose_visualizer(verbose=verbose)
        self.point_cloud_visualizer = create_point_cloud_visualizer(verbose=verbose)
        
        if self.verbose:
            print(f"FrameMatcher initialized with {orb_features} ORB features")
            print(f"Camera intrinsics: fx={camera_intrinsics['fx']:.1f}, fy={camera_intrinsics['fy']:.1f}")
    
    def match_frames(self, color_image1: np.ndarray, depth_image1: np.ndarray,
                    color_image2: np.ndarray, depth_image2: np.ndarray,
                    visualize: bool = False) -> Dict[str, Any]:
        """
        Match two RGB-depth frames and combine their point clouds.
        
        Args:
            color_image1: First RGB image (H, W, 3)
            depth_image1: First depth image (H, W), already in meters
            color_image2: Second RGB image (H, W, 3)  
            depth_image2: Second depth image (H, W), already in meters
            visualize: Whether to create visualizations (default: False)
            
        Returns:
            Dictionary containing:
            - combined_point_cloud: Combined Open3D point cloud
            - pose_estimation: Pose estimation results
            - feature_matching: Feature matching results
            - visualizations: Visualization data (if visualize=True)
        """
        if self.verbose:
            print("=== Starting Frame Matching Pipeline ===")
        
        # Step 1: Convert to grayscale for feature extraction
        gray_image1 = cv2.cvtColor(color_image1, cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(color_image2, cv2.COLOR_BGR2GRAY)
        
        # Step 2: Extract features
        if self.verbose:
            print("Extracting ORB features...")
        keypoints1, descriptors1 = self.orb_extractor.extract_features(gray_image1)
        keypoints2, descriptors2 = self.orb_extractor.extract_features(gray_image2)
        
        if self.verbose:
            print(f"Frame 1: {len(keypoints1)} keypoints, Frame 2: {len(keypoints2)} keypoints")
        
        # Step 3: Match features
        if self.verbose:
            print("Matching features...")
        matches, good_matches = self.orb_matcher.match_features(descriptors1, descriptors2)
        
        if len(good_matches) < 10:
            raise ValueError(f"Insufficient good matches: {len(good_matches)}. Need at least 10.")
        
        # Step 4: Extract matched points and compute fundamental matrix
        pts1, pts2 = self.orb_matcher.extract_matched_points(keypoints1, keypoints2, good_matches)
        F, mask = self.orb_matcher.compute_fundamental_matrix(
            pts1, pts2, method=cv2.FM_RANSAC, ransac_threshold=3.0, confidence=0.99
        )
        
        if F is None:
            raise ValueError("Failed to compute fundamental matrix")
        
        # Step 5: Estimate absolute pose
        if self.verbose:
            print("Estimating camera pose with absolute scale...")
        R_abs, t_abs, points_3d_abs, scale_factor = estimate_absolute_pose_realsense(
            F, self.K, pts1, pts2, depth_image1, depth_image2, mask, verbose=self.verbose
        )
        
        # Calculate motion metrics
        motion_metrics = calculate_motion_metrics(R_abs, t_abs, verbose=self.verbose)
        euler_angles = rotation_matrix_to_euler_angles(R_abs, order='XYZ', degrees=True, verbose=self.verbose)
        
        # Step 6: Create point clouds
        if self.verbose:
            print("Creating point clouds...")
        
        # Create point cloud from first frame
        pcd1 = self.point_cloud_visualizer.create_point_cloud_from_rgbd(
            color_image1, depth_image1, self.camera_intrinsics, depth_scale=self.depth_scale
        )
        
        # Create point cloud from second frame
        pcd2 = self.point_cloud_visualizer.create_point_cloud_from_rgbd(
            color_image2, depth_image2, self.camera_intrinsics, depth_scale=self.depth_scale
        )
        
        # Step 7: Transform second point cloud to first frame coordinates
        if self.verbose:
            print("Transforming second point cloud...")
        
        # Create inverse transformation matrix
        R_inv = R_abs.T
        t_inv = -R_inv @ t_abs
        T_inv = np.eye(4)
        T_inv[:3, :3] = R_inv
        T_inv[:3, 3] = t_inv.flatten()
        
        # Transform second point cloud
        pcd2_transformed = o3d.geometry.PointCloud(pcd2)
        pcd2_transformed.transform(T_inv)
        
        # Step 8: Combine point clouds
        if self.verbose:
            print("Combining point clouds...")
        combined_pcd = self.point_cloud_visualizer.combine_point_clouds(pcd1, pcd2_transformed)
        
        # Downsample for performance
        combined_pcd_downsampled = self.point_cloud_visualizer.downsample_point_cloud(
            combined_pcd, voxel_size=0.003, max_points=16000
        )
        
        # Prepare results
        results = {
            'combined_point_cloud': combined_pcd_downsampled,
            'pose_estimation': {
                'rotation_matrix': R_abs,
                'translation_vector': t_abs,
                'euler_angles': euler_angles,
                'scale_factor': scale_factor,
                'motion_metrics': motion_metrics,
                'reconstructed_3d_points': points_3d_abs
            },
            'feature_matching': {
                'keypoints1': keypoints1,
                'keypoints2': keypoints2,
                'good_matches': good_matches,
                'fundamental_matrix': F,
                'matched_points1': pts1,
                'matched_points2': pts2,
                'match_quality': self.orb_matcher.analyze_match_quality(good_matches),
                'match_stats': self.orb_matcher.get_match_statistics(matches, good_matches)
            }
        }
        
        # Step 9: Create visualizations if requested
        if visualize:
            if self.verbose:
                print("Creating visualizations...")
            results['visualizations'] = self._create_visualizations(
                color_image1, color_image2, keypoints1, keypoints2, good_matches,
                pcd1, pcd2_transformed, combined_pcd_downsampled, 
                R_abs, t_abs, points_3d_abs, scale_factor
            )
        
        if self.verbose:
            print(f"=== Frame Matching Complete ===")
            print(f"Combined point cloud: {len(combined_pcd_downsampled.points)} points")
            print(f"Camera movement: {np.linalg.norm(t_abs):.4f}m")
            print(f"Good matches: {len(good_matches)}")
        
        return results
    
    def _create_visualizations(self, color_image1: np.ndarray, color_image2: np.ndarray,
                             keypoints1: List, keypoints2: List, good_matches: List,
                             pcd1: o3d.geometry.PointCloud, pcd2_transformed: o3d.geometry.PointCloud,
                             combined_pcd: o3d.geometry.PointCloud,
                             R_abs: np.ndarray, t_abs: np.ndarray, points_3d_abs: np.ndarray,
                             scale_factor: float) -> Dict[str, Any]:
        """
        Create visualization data and images.
        
        Returns:
            Dictionary containing visualization data
        """
        visualizations = {}
        
        # Create keypoint visualization
        img1_with_keypoints = self.orb_matcher.visualize_keypoints(color_image1, keypoints1)
        img2_with_keypoints = self.orb_matcher.visualize_keypoints(color_image2, keypoints2)
        
        # Create matches visualization
        img_matches = self.orb_matcher.visualize_matches(
            color_image1, keypoints1, color_image2, keypoints2, good_matches, max_matches=50
        )
        
        # Create depth visualization data
        depth_viz_data = self.point_cloud_visualizer.visualize_depth_images(
            color_image1, color_image2,
            titles=("Frame 1 Depth", "Frame 2 Depth")
        )
        
        # Create reconstructed point cloud for visualization
        reconstructed_pcd = None
        if len(points_3d_abs) > 0:
            reconstructed_pcd = self.point_cloud_visualizer.create_reconstructed_point_cloud(
                points_3d_abs, color=[1, 0, 0]
            )
        
        visualizations = {
            'keypoints_img1': img1_with_keypoints,
            'keypoints_img2': img2_with_keypoints,
            'matches_image': img_matches,
            'depth_visualization': depth_viz_data,
            'individual_point_clouds': {
                'frame1_pcd': pcd1,
                'frame2_pcd': pcd2_transformed,
                'reconstructed_pcd': reconstructed_pcd
            }
        }
        
        return visualizations
    
    def visualize_3d_scene(self, results: Dict[str, Any], window_size: Tuple[int, int] = (1200, 900)) -> Any:
        """
        Create comprehensive 3D visualization using pose visualizer.
        
        Args:
            results: Results from match_frames()
            window_size: Visualization window size
            
        Returns:
            Visualization figure
        """
        pose_data = results['pose_estimation']
        
        return self.pose_visualizer.visualize_absolute_pose(
            pose_data['rotation_matrix'],
            pose_data['translation_vector'],
            points_3d=pose_data['reconstructed_3d_points'],
            origin_pcd=results['combined_point_cloud'],
            title="ED-SLAM: Frame Matching Results",
            window_size=window_size
        )
    
    def visualize_3d_scene_realtime(self, results: Dict[str, Any], 
                                   window_name: str = "ED-SLAM Real-time Point Cloud") -> None:
        """
        Create real-time 3D visualization using Open3D interactive window.
        
        Args:
            results: Results from match_frames()
            window_name: Window title for visualization
        """
        if self.verbose:
            print("Creating real-time 3D visualization with Open3D...")
        
        # Get point cloud data
        combined_pcd = results['combined_point_cloud']
        pose_data = results['pose_estimation']
        
        # Create geometries list
        geometries = []
        
        # Add combined point cloud
        if len(combined_pcd.points) > 0:
            geometries.append(combined_pcd)
            if self.verbose:
                print(f"Added combined point cloud with {len(combined_pcd.points)} points")
        
        # Add reconstructed feature points as spheres
        if len(pose_data['reconstructed_3d_points']) > 0:
            feature_spheres = self.point_cloud_visualizer.create_feature_point_spheres(
                pose_data['reconstructed_3d_points'],
                scale=0.02,
                color=[1, 0, 0],  # Red
                max_spheres=50
            )
            geometries.extend(feature_spheres)
            if self.verbose:
                print(f"Added {len(feature_spheres)} feature point spheres")
        
        # Add coordinate frame at origin
        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        geometries.append(origin_frame)
        
        # Add coordinate frame at estimated camera pose
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        # Transform camera frame to estimated pose
        T_camera = np.eye(4)
        T_camera[:3, :3] = pose_data['rotation_matrix']
        T_camera[:3, 3] = pose_data['translation_vector'].flatten()
        camera_frame.transform(T_camera)
        camera_frame.paint_uniform_color([0, 1, 0])  # Green
        geometries.append(camera_frame)
        
        if self.verbose:
            print(f"Total geometries to visualize: {len(geometries)}")
            print(f"Camera translation: {pose_data['translation_vector'].flatten()}")
            print(f"Movement distance: {np.linalg.norm(pose_data['translation_vector']):.4f}m")
        
        # Create and show visualization
        try:
            o3d.visualization.draw_geometries(
                geometries,
                window_name=window_name,
                width=1200,
                height=900,
                left=50,
                top=50
            )
        except Exception as e:
            if self.verbose:
                print(f"Visualization error: {e}")
            print("Make sure you have a display available for Open3D visualization")


class RealtimePointCloudVisualizer:
    """
    Realtime point cloud visualizer for continuous SLAM visualization.
    
    This class maintains an Open3D visualizer window that can be updated
    with new point cloud data in real-time.
    """
    
    def __init__(self, window_name: str = "ED-SLAM Real-time Point Cloud", 
                 window_size: Tuple[int, int] = (1200, 900),
                 verbose: bool = False):
        """
        Initialize realtime visualizer.
        
        Args:
            window_name: Window title
            window_size: Window dimensions (width, height)
            verbose: Enable verbose output
        """
        self.window_name = window_name
        self.window_size = window_size
        self.verbose = verbose
        
        # Initialize Open3D visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name=window_name,
            width=window_size[0],
            height=window_size[1]
        )
        
        # Set up render options
        render_option = self.vis.get_render_option()
        render_option.background_color = np.array([0.1, 0.1, 0.1])  # Dark gray
        render_option.point_size = 2.0
        
        # Geometry tracking
        self.current_geometries = {}
        self.frame_count = 0
        
        # Add coordinate frame at origin
        self.origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        self.vis.add_geometry(self.origin_frame)
        self.current_geometries['origin'] = self.origin_frame
        
        if self.verbose:
            print(f"Realtime visualizer initialized: {window_name}")
    
    def update_point_cloud(self, combined_pcd: o3d.geometry.PointCloud, 
                          pose_data: Dict[str, Any]) -> bool:
        """
        Update visualization with new point cloud data.
        
        Args:
            combined_pcd: Updated combined point cloud
            pose_data: Camera pose data
            
        Returns:
            True if visualization should continue, False to exit
        """
        try:
            # Remove old point cloud if exists
            if 'combined_pcd' in self.current_geometries:
                self.vis.remove_geometry(self.current_geometries['combined_pcd'], reset_bounding_box=False)
            
            # Add new point cloud
            if len(combined_pcd.points) > 0:
                self.vis.add_geometry(combined_pcd, reset_bounding_box=False)
                self.current_geometries['combined_pcd'] = combined_pcd
            
            # Update camera pose frame
            if 'camera_frame' in self.current_geometries:
                self.vis.remove_geometry(self.current_geometries['camera_frame'], reset_bounding_box=False)
            
            # Create new camera frame at estimated pose
            camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            T_camera = np.eye(4)
            T_camera[:3, :3] = pose_data['rotation_matrix']
            T_camera[:3, 3] = pose_data['translation_vector'].flatten()
            camera_frame.transform(T_camera)
            camera_frame.paint_uniform_color([0, 1, 0])  # Green
            
            self.vis.add_geometry(camera_frame, reset_bounding_box=False)
            self.current_geometries['camera_frame'] = camera_frame
            
            # Update feature points as spheres (limit for performance)
            if 'feature_spheres' in self.current_geometries:
                for sphere in self.current_geometries['feature_spheres']:
                    self.vis.remove_geometry(sphere, reset_bounding_box=False)
            
            if len(pose_data['reconstructed_3d_points']) > 0:
                # Create feature point spheres
                feature_spheres = []
                finite_mask = np.all(np.isfinite(pose_data['reconstructed_3d_points']), axis=1)
                valid_points = pose_data['reconstructed_3d_points'][finite_mask]
                
                # Sample points for performance
                max_spheres = 20
                step = max(1, len(valid_points) // max_spheres)
                
                for i, point in enumerate(valid_points[::step]):
                    if i >= max_spheres:
                        break
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
                    sphere.translate(point)
                    sphere.paint_uniform_color([1, 0, 0])  # Red
                    feature_spheres.append(sphere)
                    self.vis.add_geometry(sphere, reset_bounding_box=False)
                
                self.current_geometries['feature_spheres'] = feature_spheres
            
            self.frame_count += 1
            
            if self.verbose and self.frame_count % 5 == 0:  # Print every 5 frames
                print(f"Frame {self.frame_count}: {len(combined_pcd.points)} points, "
                      f"movement: {np.linalg.norm(pose_data['translation_vector']):.3f}m")
            
            # Update visualization
            self.vis.poll_events()
            self.vis.update_renderer()
            
            return not self.vis.should_close()
            
        except Exception as e:
            if self.verbose:
                print(f"Visualization update error: {e}")
            return False
    
    def close(self):
        """Close the visualization window."""
        try:
            self.vis.destroy_window()
            if self.verbose:
                print("Realtime visualizer closed")
        except:
            pass
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()


    def save_results(self, results: Dict[str, Any], output_dir: str) -> bool:
        """
        Save frame matching results to files.
        
        Args:
            results: Results from match_frames()
            output_dir: Output directory path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Save combined point cloud
            self.point_cloud_visualizer.save_point_cloud(
                results['combined_point_cloud'],
                str(output_path / "combined_pointcloud.ply")
            )
            
            # Save pose and feature data
            pose_data = results['pose_estimation']
            feature_data = results['feature_matching']
            
            np.savez(str(output_path / "frame_matching_data.npz"),
                     rotation_matrix=pose_data['rotation_matrix'],
                     translation_vector=pose_data['translation_vector'],
                     euler_angles=pose_data['euler_angles'],
                     scale_factor=pose_data['scale_factor'],
                     reconstructed_3d_points=pose_data['reconstructed_3d_points'],
                     fundamental_matrix=feature_data['fundamental_matrix'],
                     keypoints1=np.array([kp.pt for kp in feature_data['keypoints1']]),
                     keypoints2=np.array([kp.pt for kp in feature_data['keypoints2']]),
                     match_distances=np.array([m.distance for m in feature_data['good_matches']]),
                     match_quality=feature_data['match_quality'],
                     match_stats=feature_data['match_stats'])
            
            # Save visualization images if available
            if 'visualizations' in results:
                viz_data = results['visualizations']
                cv2.imwrite(str(output_path / "keypoints_frame1.jpg"), viz_data['keypoints_img1'])
                cv2.imwrite(str(output_path / "keypoints_frame2.jpg"), viz_data['keypoints_img2'])
                cv2.imwrite(str(output_path / "feature_matches.jpg"), viz_data['matches_image'])
                
                # Save individual point clouds
                if viz_data['individual_point_clouds']['frame1_pcd'] is not None:
                    self.point_cloud_visualizer.save_point_cloud(
                        viz_data['individual_point_clouds']['frame1_pcd'],
                        str(output_path / "frame1_pointcloud.ply")
                    )
                
                if viz_data['individual_point_clouds']['frame2_pcd'] is not None:
                    self.point_cloud_visualizer.save_point_cloud(
                        viz_data['individual_point_clouds']['frame2_pcd'],
                        str(output_path / "frame2_pointcloud.ply")
                    )
                
                if viz_data['individual_point_clouds']['reconstructed_pcd'] is not None:
                    self.point_cloud_visualizer.save_point_cloud(
                        viz_data['individual_point_clouds']['reconstructed_pcd'],
                        str(output_path / "reconstructed_points.ply")
                    )
            
            if self.verbose:
                print(f"Results saved to {output_dir}")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"Error saving results: {e}")
            return False


def create_frame_matcher(camera_intrinsics: Dict[str, float], depth_scale: float = 0.001,
                        orb_features: int = 500, match_distance_threshold: float = 50.0,
                        verbose: bool = False) -> FrameMatcher:
    """
    Factory function to create a FrameMatcher instance.
    
    Args:
        camera_intrinsics: Dictionary with fx, fy, cx, cy values
        depth_scale: Depth scale factor (default: 0.001)
        orb_features: Number of ORB features (default: 500)
        match_distance_threshold: Feature matching threshold (default: 50.0)
        verbose: Enable verbose output (default: False)
        
    Returns:
        FrameMatcher instance
    """
    return FrameMatcher(
        camera_intrinsics=camera_intrinsics,
        depth_scale=depth_scale,
        orb_features=orb_features,
        match_distance_threshold=match_distance_threshold,
        verbose=verbose
    )