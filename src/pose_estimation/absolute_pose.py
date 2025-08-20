"""
Absolute pose estimation using RealSense depth information

This module provides functions for estimating camera pose with absolute scale
by leveraging depth information from RealSense cameras.
"""

import numpy as np
import cv2


def estimate_absolute_pose_realsense(
    F, K, pts1, pts2, depth1, depth2, mask, verbose=False
):
    """
    Absolute scale pose estimation using RealSense depth information

    Args:
        F: Fundamental matrix (3x3)
        K: Camera intrinsic matrix (RGB camera)
        pts1, pts2: Corresponding points (Nx2)
        depth1, depth2: Depth images (already scaled to meters)
        mask: Valid matching mask
        verbose: Enable verbose output (default: False)

    Returns:
        R: Rotation matrix (3x3)
        t_absolute: Absolute scale translation vector (3x1)
        points_3d_absolute: Absolute scale 3D points
        scale_factor: Calculated scale factor
    """

    # 1. Basic relative pose estimation
    E = K.T @ F @ K
    points_count, R, t_relative, pose_mask = cv2.recoverPose(E, pts1, pts2, K)

    if verbose:
        print("=== Step 1: Relative pose estimation completed ===")
        print(f"Relative translation vector: {t_relative.flatten()}")
        print(f"Relative translation magnitude: {np.linalg.norm(t_relative):.4f}")

    # 2. Collect actual depth information for valid matching points
    valid_indices = (mask.flatten() > 0) & (pose_mask.flatten() > 0)
    valid_pts1 = pts1[valid_indices]
    valid_pts2 = pts2[valid_indices]

    real_depths_1 = []
    real_depths_2 = []
    valid_3d_indices = []

    if verbose:
        print("=== Step 2: Collecting actual depth information ===")

    for i, (pt1, pt2) in enumerate(zip(valid_pts1, valid_pts2)):
        # Extract point coordinates - handle different array formats
        if len(pt1.shape) == 2:  # (1, 2) format
            x1, y1 = int(pt1[0, 0]), int(pt1[0, 1])
            x2, y2 = int(pt2[0, 0]), int(pt2[0, 1])
        else:  # (2,) format
            x1, y1 = int(pt1[0]), int(pt1[1])
            x2, y2 = int(pt2[0]), int(pt2[1])

        # Depth from first image (already in meters)
        if 0 <= x1 < depth1.shape[1] and 0 <= y1 < depth1.shape[0]:
            depth_val_1 = depth1[y1, x1]

            # Depth from second image (already in meters)
            if 0 <= x2 < depth2.shape[1] and 0 <= y2 < depth2.shape[0]:
                depth_val_2 = depth2[y2, x2]

                # Use only when both have valid depth values
                if depth_val_1 > 0.1 and depth_val_2 > 0.1:  # More than 10cm
                    real_depths_1.append(depth_val_1)
                    real_depths_2.append(depth_val_2)
                    valid_3d_indices.append(i)

    if verbose:
        print(f"Points with valid depth information: {len(real_depths_1)}")

    if len(real_depths_1) < 5:
        if verbose:
            print("Warning: Insufficient valid depth information!")
        return R, t_relative, np.array([]), 1.0

    # 3. Relative 3D reconstruction
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t_relative])

    # Convert point format to (2, N)
    if len(valid_pts1.shape) == 3:  # (N, 1, 2) format
        pts1_2d = valid_pts1.squeeze(1).T  # (2, N)
        pts2_2d = valid_pts2.squeeze(1).T  # (2, N)
    else:  # (N, 2) format
        pts1_2d = valid_pts1.T
        pts2_2d = valid_pts2.T

    points_4d = cv2.triangulatePoints(P1, P2, pts1_2d, pts2_2d)
    points_3d_relative = (points_4d[:3] / points_4d[3]).T

    # 4. Scale factor calculation
    if verbose:
        print("=== Step 3: Absolute scale calculation ===")

    # Method A: Depth comparison based on first camera
    reconstructed_depths_1 = np.abs(
        points_3d_relative[valid_3d_indices, 2]
    )  # Z coordinate
    scale_factors_1 = np.array(real_depths_1) / reconstructed_depths_1

    # Outlier removal (IQR method)
    Q1 = np.percentile(scale_factors_1, 25)
    Q3 = np.percentile(scale_factors_1, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    valid_scale_mask = (scale_factors_1 >= lower_bound) & (
        scale_factors_1 <= upper_bound
    )
    filtered_scale_factors = scale_factors_1[valid_scale_mask]

    if len(filtered_scale_factors) > 0:
        scale_factor = np.median(filtered_scale_factors)
        scale_std = np.std(filtered_scale_factors)

        if verbose:
            print(f"Calculated scale factor: {scale_factor:.4f} ± {scale_std:.4f}")
            print(f"Points used: {len(filtered_scale_factors)}/{len(scale_factors_1)}")

        # 5. Apply absolute scale
        t_absolute = t_relative * scale_factor
        points_3d_absolute = points_3d_relative * scale_factor

        if verbose:
            print("=== Step 4: Absolute scale results ===")
            print(
                f"Absolute camera movement distance: {np.linalg.norm(t_absolute):.4f}m"
            )
            print(f"Absolute translation vector: {t_absolute.flatten()}")

        # 6. Validation: Compare actual depth with reconstructed depth
        scaled_depths_1 = reconstructed_depths_1 * scale_factor
        depth_errors = np.abs(np.array(real_depths_1) - scaled_depths_1)
        avg_depth_error = np.mean(depth_errors)

        if verbose:
            print(f"Depth reconstruction error: {avg_depth_error:.4f}m (average)")
            print(
                f"Depth reconstruction accuracy: {(1 - avg_depth_error/np.mean(real_depths_1))*100:.1f}%"
            )

        return R, t_absolute, points_3d_absolute, scale_factor

    else:
        if verbose:
            print("Error: Unable to calculate valid scale factor!")
        return R, t_relative, points_3d_relative, 1.0


def calculate_motion_metrics(R, t_absolute, verbose=False):
    """
    Calculate motion metrics based on absolute scale

    Args:
        R: Rotation matrix (3x3)
        t_absolute: Absolute scale translation vector (3x1)
        verbose: Enable verbose output (default: False)

    Returns:
        dict: Motion metric information
            - distance: Translation distance (m)
            - direction: Normalized translation direction vector
            - rotation_angle: Rotation angle (degrees)
            - primary_axis: Primary movement axis ('X', 'Y', 'Z')
    """
    # Translation distance
    translation_distance = np.linalg.norm(t_absolute)

    # Rotation angle (Rodrigues vector)
    rotation_vector, _ = cv2.Rodrigues(R)
    rotation_angle = np.linalg.norm(rotation_vector) * 180 / np.pi

    # Translation direction
    if translation_distance > 0:
        translation_direction = t_absolute.flatten() / translation_distance

        # Determine primary movement axis
        max_axis = np.argmax(np.abs(translation_direction))
        axis_names = ["X", "Y", "Z"]
        primary_direction = axis_names[max_axis]

        if verbose:
            print(f"\n=== Absolute Motion Analysis ===")
            print(
                f"Total translation distance: {translation_distance:.4f}m ({translation_distance*100:.1f}cm)"
            )
            print(
                f"Primary movement direction: {primary_direction}-axis ({translation_direction[max_axis]:+.3f})"
            )
            print(f"Total rotation angle: {rotation_angle:.2f} degrees")

        return {
            "distance": translation_distance,
            "direction": translation_direction,
            "rotation_angle": rotation_angle,
            "primary_axis": primary_direction,
        }

    return None
