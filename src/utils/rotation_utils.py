"""
Rotation utilities for SLAM system

This module provides utilities for rotation matrix conversions and manipulations,
including conversion between rotation matrices and Euler angles.
"""

import numpy as np
import cv2
from typing import Tuple, Union


def rotation_matrix_to_euler_angles(
    R: np.ndarray, order: str = "XYZ", degrees: bool = True, verbose: bool = False
) -> np.ndarray:
    """
    Convert rotation matrix to Euler angles.

    This function converts a 3x3 rotation matrix to Euler angles using the specified
    rotation order. The default implementation uses XYZ (Roll-Pitch-Yaw) convention.

    Args:
        R: 3x3 rotation matrix
        order: Rotation order ('XYZ', 'ZYX', etc.) - default: 'XYZ'
        degrees: Return angles in degrees if True, radians if False (default: True)
        verbose: Enable verbose output (default: False)

    Returns:
        np.ndarray: Euler angles [roll, pitch, yaw] or [x, y, z] depending on convention

    Raises:
        ValueError: If rotation matrix is not valid
        NotImplementedError: If rotation order is not supported
    """
    # Validate input
    if R.shape != (3, 3):
        raise ValueError(f"Expected 3x3 rotation matrix, got shape {R.shape}")

    # Check if matrix is approximately a valid rotation matrix
    if not _is_valid_rotation_matrix(R):
        if verbose:
            print("Warning: Input matrix may not be a valid rotation matrix")

    if order.upper() == "XYZ":
        # XYZ rotation order (Roll-Pitch-Yaw)
        euler_angles = _rotation_matrix_to_xyz_euler(R)
    elif order.upper() == "ZYX":
        # ZYX rotation order (Yaw-Pitch-Roll)
        euler_angles = _rotation_matrix_to_zyx_euler(R)
    else:
        raise NotImplementedError(f"Rotation order '{order}' is not implemented")

    # Convert to degrees if requested
    if degrees:
        euler_angles = euler_angles * 180.0 / np.pi

    if verbose:
        unit = "degrees" if degrees else "radians"
        if order.upper() == "XYZ":
            print(
                f"Euler angles ({order}) in {unit}: Roll={euler_angles[0]:.3f}, Pitch={euler_angles[1]:.3f}, Yaw={euler_angles[2]:.3f}"
            )
        else:
            print(
                f"Euler angles ({order}) in {unit}: [{euler_angles[0]:.3f}, {euler_angles[1]:.3f}, {euler_angles[2]:.3f}]"
            )

    return euler_angles


def _rotation_matrix_to_xyz_euler(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to XYZ Euler angles (Roll-Pitch-Yaw).

    This is the implementation used in the notebook, with improved singularity handling.

    Args:
        R: 3x3 rotation matrix

    Returns:
        np.ndarray: [roll, pitch, yaw] in radians
    """
    # Extract rotation angles
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    # Check for singularity (gimbal lock)
    singular = sy < 1e-6

    if not singular:
        # Normal case
        roll = np.arctan2(R[2, 1], R[2, 2])  # Rotation around X-axis
        pitch = np.arctan2(-R[2, 0], sy)  # Rotation around Y-axis
        yaw = np.arctan2(R[1, 0], R[0, 0])  # Rotation around Z-axis
    else:
        # Singularity case (gimbal lock)
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0  # Set yaw to 0 in singular case

    return np.array([roll, pitch, yaw])


def _rotation_matrix_to_zyx_euler(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to ZYX Euler angles (Yaw-Pitch-Roll).

    Args:
        R: 3x3 rotation matrix

    Returns:
        np.ndarray: [yaw, pitch, roll] in radians
    """
    # Extract rotation angles for ZYX order
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        pitch = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[2, 1], R[2, 2])
    else:
        yaw = np.arctan2(-R[0, 1], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        roll = 0

    return np.array([yaw, pitch, roll])


def euler_angles_to_rotation_matrix(
    euler_angles: np.ndarray,
    order: str = "XYZ",
    degrees: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """
    Convert Euler angles to rotation matrix.

    Args:
        euler_angles: Euler angles array [angle1, angle2, angle3]
        order: Rotation order ('XYZ', 'ZYX') - default: 'XYZ'
        degrees: Input angles in degrees if True, radians if False (default: True)
        verbose: Enable verbose output (default: False)

    Returns:
        np.ndarray: 3x3 rotation matrix

    Raises:
        NotImplementedError: If rotation order is not supported
    """
    angles = np.array(euler_angles, dtype=np.float64)

    # Convert to radians if necessary
    if degrees:
        angles = angles * np.pi / 180.0

    if order.upper() == "XYZ":
        R = _xyz_euler_to_rotation_matrix(angles)
    elif order.upper() == "ZYX":
        R = _zyx_euler_to_rotation_matrix(angles)
    else:
        raise NotImplementedError(f"Rotation order '{order}' is not implemented")

    if verbose:
        unit = "degrees" if degrees else "radians"
        print(f"Converted Euler angles {euler_angles} ({unit}) to rotation matrix:")
        print(R)

    return R


def _xyz_euler_to_rotation_matrix(angles: np.ndarray) -> np.ndarray:
    """Convert XYZ Euler angles to rotation matrix."""
    roll, pitch, yaw = angles

    # Individual rotation matrices
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    )

    Ry = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )

    Rz = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )

    # Combined rotation: R = Rz * Ry * Rx
    return Rz @ Ry @ Rx


def _zyx_euler_to_rotation_matrix(angles: np.ndarray) -> np.ndarray:
    """Convert ZYX Euler angles to rotation matrix."""
    yaw, pitch, roll = angles

    # Individual rotation matrices
    Rz = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )

    Ry = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )

    Rx = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    )

    # Combined rotation: R = Rx * Ry * Rz
    return Rx @ Ry @ Rz


def rodrigues_to_rotation_matrix(
    rodrigues_vector: np.ndarray, verbose: bool = False
) -> np.ndarray:
    """
    Convert Rodrigues vector to rotation matrix using OpenCV.

    Args:
        rodrigues_vector: 3D Rodrigues rotation vector
        verbose: Enable verbose output (default: False)

    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    R, _ = cv2.Rodrigues(rodrigues_vector)

    if verbose:
        angle = np.linalg.norm(rodrigues_vector)
        print(
            f"Rodrigues vector {rodrigues_vector} (angle: {angle:.4f} rad) -> rotation matrix"
        )

    return R


def rotation_matrix_to_rodrigues(R: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Convert rotation matrix to Rodrigues vector using OpenCV.

    Args:
        R: 3x3 rotation matrix
        verbose: Enable verbose output (default: False)

    Returns:
        np.ndarray: 3D Rodrigues rotation vector
    """
    rodrigues_vector, _ = cv2.Rodrigues(R)
    rodrigues_vector = rodrigues_vector.flatten()

    if verbose:
        angle = np.linalg.norm(rodrigues_vector)
        print(
            f"Rotation matrix -> Rodrigues vector {rodrigues_vector} (angle: {angle:.4f} rad)"
        )

    return rodrigues_vector


def rotation_angle_from_matrix(R: np.ndarray, verbose: bool = False) -> float:
    """
    Compute the rotation angle from a rotation matrix.

    Args:
        R: 3x3 rotation matrix
        verbose: Enable verbose output (default: False)

    Returns:
        float: Rotation angle in radians
    """
    # Use trace to compute rotation angle
    trace = np.trace(R)
    # Clamp trace to valid range to avoid numerical issues
    trace = np.clip(trace, -1, 3)
    angle = np.arccos((trace - 1) / 2)

    if verbose:
        print(f"Rotation angle: {angle:.4f} rad ({angle * 180/np.pi:.2f} degrees)")

    return angle


def _is_valid_rotation_matrix(R: np.ndarray, tolerance: float = 1e-6) -> bool:
    """
    Check if a matrix is a valid rotation matrix.

    A valid rotation matrix should satisfy:
    1. R^T * R = I (orthogonality)
    2. det(R) = 1 (proper rotation, not reflection)

    Args:
        R: 3x3 matrix to check
        tolerance: Numerical tolerance for checks

    Returns:
        bool: True if matrix is a valid rotation matrix
    """
    if R.shape != (3, 3):
        return False

    # Check orthogonality: R^T * R should be identity
    should_be_identity = R.T @ R
    identity = np.eye(3)
    if not np.allclose(should_be_identity, identity, atol=tolerance):
        return False

    # Check determinant: should be 1
    det = np.linalg.det(R)
    if not np.isclose(det, 1.0, atol=tolerance):
        return False

    return True


def compose_rotations(
    R1: np.ndarray, R2: np.ndarray, verbose: bool = False
) -> np.ndarray:
    """
    Compose two rotation matrices.

    Args:
        R1: First rotation matrix (3x3)
        R2: Second rotation matrix (3x3)
        verbose: Enable verbose output (default: False)

    Returns:
        np.ndarray: Composed rotation matrix R2 * R1
    """
    R_composed = R2 @ R1

    if verbose:
        angle1 = rotation_angle_from_matrix(R1) * 180 / np.pi
        angle2 = rotation_angle_from_matrix(R2) * 180 / np.pi
        angle_composed = rotation_angle_from_matrix(R_composed) * 180 / np.pi
        print(
            f"Composed rotations: {angle1:.2f}° + {angle2:.2f}° = {angle_composed:.2f}°"
        )

    return R_composed


def invert_rotation(R: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Invert a rotation matrix (equivalent to transpose for rotation matrices).

    Args:
        R: 3x3 rotation matrix
        verbose: Enable verbose output (default: False)

    Returns:
        np.ndarray: Inverted rotation matrix R^T
    """
    R_inv = R.T

    if verbose:
        print("Inverted rotation matrix (R^T)")

    return R_inv
