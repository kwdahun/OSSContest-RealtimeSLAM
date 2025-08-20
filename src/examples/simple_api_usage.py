"""
Simple API Usage Example

This demonstrates the clean, high-level API for frame matching.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from slam.frame_matcher import create_frame_matcher


def match_two_frames(color_image1, depth_image1, color_image2, depth_image2, 
                    camera_intrinsics, depth_scale=0.001, visualize=False):
    """
    High-level function to match two RGB-depth frames.
    
    Args:
        color_image1, depth_image1: First RGB-depth frame pair
        color_image2, depth_image2: Second RGB-depth frame pair  
        camera_intrinsics: Dict with fx, fy, cx, cy
        depth_scale: Depth scale factor (default: 0.001)
        visualize: Whether to create visualizations (default: False)
        
    Returns:
        Dictionary with combined_point_cloud, pose_estimation, feature_matching, etc.
    """
    
    # Create frame matcher
    matcher = create_frame_matcher(
        camera_intrinsics=camera_intrinsics,
        depth_scale=depth_scale,
        verbose=True
    )
    
    # Match frames and return combined point cloud + results
    return matcher.match_frames(
        color_image1, depth_image1,
        color_image2, depth_image2,
        visualize=visualize
    )


# Example usage:
"""
# Assuming you have RGB-depth frames from RealSense or other source:

camera_intrinsics = {
    'fx': 605.13,
    'fy': 604.74, 
    'cx': 330.82,
    'cy': 251.82
}

# Match frames and get combined point cloud
results = match_two_frames(
    color_img1, depth_img1,  # First frame
    color_img2, depth_img2,  # Second frame
    camera_intrinsics,
    visualize=True
)

# Access results
combined_pcd = results['combined_point_cloud']
pose_info = results['pose_estimation']  
matches_info = results['feature_matching']

# The combined point cloud now contains points from both frames
# aligned in the first frame's coordinate system
"""