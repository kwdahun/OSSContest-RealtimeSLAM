"""
Realtime Point Cloud Visualization Example

This script demonstrates continuous real-time point cloud visualization
using Open3D's interactive window. It captures multiple frames and updates
the visualization continuously.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from sensors.realsense_camera import RealSenseD435i
from slam.frame_matcher import create_frame_matcher, RealtimePointCloudVisualizer


def main():
    """
    Continuous real-time point cloud visualization example.
    """
    print("=== ED-SLAM Real-time Visualization Example ===")
    
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
    
    # Create FrameMatcher
    frame_matcher = create_frame_matcher(
        camera_intrinsics=camera_intrinsics,
        depth_scale=camera.depth_scale,
        orb_features=500,
        match_distance_threshold=50.0,
        verbose=True
    )
    
    # Initialize realtime visualizer
    realtime_viz = RealtimePointCloudVisualizer(
        window_name="ED-SLAM Continuous Real-time Visualization",
        verbose=True
    )
    
    try:
        # Capture reference frame
        print("\nCapturing reference frame...")
        ref_pair = camera.capture_frame_pair()
        ref_color, ref_depth = ref_pair.get("rgb"), ref_pair.get("depth")
        
        print("\nStarting continuous visualization...")
        print("Move the camera around to see real-time point cloud updates")
        print("Close the visualization window or press Ctrl+C to stop")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            try:
                # Capture new frame
                current_pair = camera.capture_frame_pair()
                current_color, current_depth = current_pair.get("rgb"), current_pair.get("depth")
                
                # Match frames
                results = frame_matcher.match_frames(
                    ref_color, ref_depth,
                    current_color, current_depth,
                    visualize=False  # Disable other visualizations for performance
                )
                
                # Update real-time visualization
                should_continue = realtime_viz.update_point_cloud(
                    results['combined_point_cloud'],
                    results['pose_estimation']
                )
                
                if not should_continue:
                    print("\nVisualization window closed")
                    break
                
                frame_count += 1
                
                # Print performance stats every 10 frames
                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"Processed {frame_count} frames, FPS: {fps:.1f}")
                
                # Update reference frame every few iterations for better tracking
                if frame_count % 5 == 0:
                    ref_color, ref_depth = current_color, current_depth
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.05)  # ~20 FPS max
                
            except KeyboardInterrupt:
                print("\nStopped by user")
                break
            except Exception as e:
                print(f"Frame processing error: {e}")
                continue
                
    except Exception as e:
        print(f"Error during continuous visualization: {e}")
    
    finally:
        # Cleanup
        realtime_viz.close()
        camera.stop()
        print("Cleanup complete")


if __name__ == "__main__":
    main()