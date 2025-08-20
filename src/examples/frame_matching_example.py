"""
Example script demonstrating the FrameMatcher API

This script shows how to use the high-level FrameMatcher API 
to match two RGB-depth frames and combine their point clouds.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from sensors.realsense_camera import RealSenseD435i
from slam.frame_matcher import create_frame_matcher


def main():
    """
    Example usage of FrameMatcher with RealSense camera.
    """
    print("=== ED-SLAM Frame Matching Example ===")
    
    # Initialize RealSense camera
    print("Initializing RealSense camera...")
    camera = RealSenseD435i()
    camera.start()
    
    # Get camera intrinsics
    camera_intrinsics = {
        'fx': camera.camera_intrinsics.fx,
        'fy': camera.camera_intrinsics.fy,
        'cx': camera.camera_intrinsics.cx,
        'cy': camera.camera_intrinsics.cy
    }
    
    # Create FrameMatcher with camera parameters
    frame_matcher = create_frame_matcher(
        camera_intrinsics=camera_intrinsics,
        depth_scale=camera.depth_scale,
        orb_features=500,
        match_distance_threshold=50.0,
        verbose=True
    )
    
    try:
        # Capture first frame
        print("\nCapturing first frame...")
        captured_pair1 = camera.capture_frame_pair()
        color_image1, depth_image1 = captured_pair1.get("rgb"), captured_pair1.get("depth")
        
        input("Press Enter to capture second frame...")
        
        # Capture second frame
        print("Capturing second frame...")
        captured_pair2 = camera.capture_frame_pair()
        color_image2, depth_image2 = captured_pair2.get("rgb"), captured_pair2.get("depth")
        
        # Match frames and combine point clouds
        print("\n=== Running Frame Matching ===")
        results = frame_matcher.match_frames(
            color_image1, depth_image1,
            color_image2, depth_image2,
            visualize=True  # Enable visualizations
        )
        
        # Print results summary
        print("\n=== Results Summary ===")
        pose_data = results['pose_estimation']
        feature_data = results['feature_matching']
        
        print(f"Combined point cloud: {len(results['combined_point_cloud'].points)} points")
        print(f"Camera movement: {np.linalg.norm(pose_data['translation_vector']):.4f}m")
        print(f"Euler angles (Roll, Pitch, Yaw): {pose_data['euler_angles']}")
        print(f"Feature matches: {len(feature_data['good_matches'])}")
        print(f"Match quality (avg distance): {feature_data['match_quality']['avg_distance']:.2f}")
        
        # Create 3D visualization (traditional plotly version)
        print("\nCreating 3D visualization (plotly)...")
        frame_matcher.visualize_3d_scene(results)
        
        # Create real-time 3D visualization (new Open3D version)
        print("\nCreating real-time 3D visualization (Open3D)...")
        frame_matcher.visualize_3d_scene_realtime(results)
        
        # Save results
        print("\nSaving results...")
        save_success = frame_matcher.save_results(results, "frame_matching_output")
        if save_success:
            print("Results saved to frame_matching_output/")
        else:
            print("Failed to save results")
        
    except Exception as e:
        print(f"Error during frame matching: {e}")
    
    finally:
        # Cleanup
        camera.stop()
        print("Camera stopped")


if __name__ == "__main__":
    main()