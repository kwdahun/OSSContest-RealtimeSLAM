"""
SLAM module for ED-SLAM system.

This module provides high-level APIs for SLAM operations including
frame matching, point cloud fusion, and pose estimation.
"""

from .frame_matcher import FrameMatcher, create_frame_matcher

__all__ = ['FrameMatcher', 'create_frame_matcher']