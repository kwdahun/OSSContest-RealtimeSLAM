import pyrealsense2 as rs
import numpy as np
import cv2
from typing import Tuple, Optional, Dict
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.camera_utils import CameraIntrinsics


class RealSenseD435i:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        """
        Intel RealSense D435i camera interface with RGB-Depth alignment.
        frame_capture returns RGB and Depth frames in meter as numpy arrays.

        Args:
            width: Frame width
            height: Frame height  
            fps: Frames per second
        """
        self.width = width
        self.height = height
        self.fps = fps
        
        # Initialize pipeline and config
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure streams
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        
        # Alignment object to align depth to color
        self.align = rs.align(rs.stream.color)
        
        # Camera intrinsics (will be set after pipeline start)
        self.color_intrinsics = None
        self.depth_intrinsics = None
        self.camera_intrinsics = None
        
        # Pipeline profile
        self.profile = None
        
        # Depth scale
        self.depth_scale = None
        
    def start(self) -> bool:
        """
        Start the camera pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Start streaming
            self.profile = self.pipeline.start(self.config)
            
            # Get device and depth sensor
            device = self.profile.get_device()
            depth_sensor = device.first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            # Get camera intrinsics
            self._get_camera_intrinsics()
            
            print(f"RealSense D435i started successfully")
            print(f"Depth scale: {self.depth_scale}")
            print(f"Color intrinsics: fx={self.color_intrinsics.fx:.1f}, fy={self.color_intrinsics.fy:.1f}")
            
            return True
            
        except Exception as e:
            print(f"Failed to start RealSense camera: {e}")
            return False
    
    def stop(self):
        """Stop the camera pipeline."""
        try:
            self.pipeline.stop()
            print("RealSense D435i stopped")
        except Exception as e:
            print(f"Error stopping camera: {e}")
    
    def get_frames(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get aligned RGB and depth frames.
        
        Returns:
            Tuple of (rgb_frame, depth_frame) or None if failed
        """
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames()
            
            # Align depth frame to color frame
            aligned_frames = self.align.process(frames)
            
            # Get aligned frames
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None
            
            # Convert to numpy arrays
            rgb_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data()) * self.depth_scale
            
            return rgb_image, depth_image
            
        except Exception as e:
            print(f"Error getting frames: {e}")
            return None
    
    def get_camera_intrinsics(self) -> Optional[CameraIntrinsics]:
        """Get camera intrinsics as CameraIntrinsics object."""
        return self.camera_intrinsics
    
    def _get_camera_intrinsics(self):
        """Extract camera intrinsics from RealSense profile."""
        try:
            # Get color stream profile
            color_profile = rs.video_stream_profile(
                self.profile.get_stream(rs.stream.color)
            )
            self.color_intrinsics = color_profile.get_intrinsics()
            
            # Get depth stream profile  
            depth_profile = rs.video_stream_profile(
                self.profile.get_stream(rs.stream.depth)
            )
            self.depth_intrinsics = depth_profile.get_intrinsics()
            
            # Create CameraIntrinsics object using color intrinsics
            # (since we align depth to color)
            self.camera_intrinsics = CameraIntrinsics(
                fx=self.color_intrinsics.fx,
                fy=self.color_intrinsics.fy,
                cx=self.color_intrinsics.ppx,
                cy=self.color_intrinsics.ppy,
                width=self.color_intrinsics.width,
                height=self.color_intrinsics.height
            )
            
        except Exception as e:
            print(f"Error getting camera intrinsics: {e}")
    
    def get_depth_scale(self) -> float:
        """Get depth scale factor."""
        return self.depth_scale if self.depth_scale else 0.001  # Default to 1mm if not set
    
    def capture_frame_pair(self) -> Optional[Dict]:
        """
        Capture a single aligned RGB-Depth frame pair.
        
        Returns:
            Dictionary with frame data and metadata
        """
        frame_data = self.get_frames()
        if frame_data is None:
            return None
            
        rgb_frame, depth_frame = frame_data
        
        return {
            'rgb': rgb_frame,
            'depth': depth_frame,
            'intrinsics': self.camera_intrinsics,
            'depth_scale': self.depth_scale,
            'timestamp': cv2.getTickCount(),
            'width': self.width,
            'height': self.height
        }


def test_camera_connection():
    """Test RealSense camera connection and display basic info."""
    try:
        # Create context and get device list
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            print("No RealSense devices found!")
            return False
        
        print(f"Found {len(devices)} RealSense device(s):")
        for i, device in enumerate(devices):
            print(f"  Device {i}: {device.get_info(rs.camera_info.name)}")
            print(f"    Serial: {device.get_info(rs.camera_info.serial_number)}")
            print(f"    Firmware: {device.get_info(rs.camera_info.firmware_version)}")
        
        return True
        
    except Exception as e:
        print(f"Error testing camera connection: {e}")
        return False


if __name__ == "__main__":
    # Test camera connection
    if test_camera_connection():
        print("\nTesting camera stream...")
        
        camera = RealSenseD435i()
        if camera.start():
            try:
                for i in range(5):
                    frame_data = camera.capture_frame_pair()
                    if frame_data:
                        print(f"Frame {i+1}: RGB {frame_data['rgb'].shape}, "
                              f"Depth {frame_data['depth'].shape}")
                    else:
                        print(f"Frame {i+1}: Failed to capture")
                        
            finally:
                camera.stop()
        else:
            print("Failed to start camera")
    else:
        print("Camera connection test failed")