"""
ORB Feature Matcher

This module provides ORB-based feature matching functionality for SLAM systems.
Based on the implementation from capture_and_match.ipynb notebook.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict


class ORBMatcher:
    """
    ORB feature matcher using BFMatcher with cross-check validation.

    This class implements feature matching between two sets of ORB descriptors
    with configurable distance thresholds and matching strategies.
    """

    def __init__(
        self,
        distance_threshold: float = 50.0,
        cross_check: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize ORB matcher.

        Args:
            distance_threshold: Maximum Hamming distance for good matches (default: 50.0)
            cross_check: Enable cross-check validation (default: True)
            verbose: Enable verbose output (default: False)
        """
        self.distance_threshold = distance_threshold
        self.cross_check = cross_check
        self.verbose = verbose

        # Create BFMatcher with Hamming distance for ORB descriptors
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=self.cross_check)

        if self.verbose:
            print(f"ORB Matcher initialized:")
            print(f"  Distance threshold: {self.distance_threshold}")
            print(f"  Cross-check: {self.cross_check}")

    def match_features(
        self, descriptors1: np.ndarray, descriptors2: np.ndarray
    ) -> Tuple[List[cv2.DMatch], List[cv2.DMatch]]:
        """
        Match features between two descriptor sets.

        Args:
            descriptors1: First set of ORB descriptors (N1 x 32)
            descriptors2: Second set of ORB descriptors (N2 x 32)

        Returns:
            Tuple containing:
            - all_matches: All matches sorted by distance
            - good_matches: Good matches filtered by distance threshold
        """
        if descriptors1 is None or descriptors2 is None:
            if self.verbose:
                print("Warning: One or both descriptor sets are None")
            return [], []

        if len(descriptors1) == 0 or len(descriptors2) == 0:
            if self.verbose:
                print("Warning: One or both descriptor sets are empty")
            return [], []

        # Match descriptors
        matches = self.bf_matcher.match(descriptors1, descriptors2)

        # Sort matches by distance (best matches first)
        matches = sorted(matches, key=lambda x: x.distance)

        if self.verbose:
            print(f"Total matches found: {len(matches)}")

        # Filter good matches by distance threshold
        good_matches = [m for m in matches if m.distance < self.distance_threshold]

        if self.verbose:
            print(
                f"Good matches (distance < {self.distance_threshold}): {len(good_matches)}"
            )

        return matches, good_matches

    def extract_matched_points(
        self,
        keypoints1: List[cv2.KeyPoint],
        keypoints2: List[cv2.KeyPoint],
        good_matches: List[cv2.DMatch],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract matched point coordinates from keypoints and matches.

        Args:
            keypoints1: Keypoints from first image
            keypoints2: Keypoints from second image
            good_matches: List of good matches

        Returns:
            Tuple of matched points:
            - pts1: Points from first image (N x 1 x 2)
            - pts2: Points from second image (N x 1 x 2)
        """
        if len(good_matches) == 0:
            if self.verbose:
                print("Warning: No good matches to extract points from")
            return np.array([]).reshape(0, 1, 2), np.array([]).reshape(0, 1, 2)

        # Extract matched keypoint coordinates
        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )

        if self.verbose:
            print(f"Extracted {len(pts1)} matched point pairs")
            if len(pts1) > 0:
                print(
                    f"Sample points from frame 1: {pts1[:min(3, len(pts1))].reshape(-1, 2)}"
                )
                print(
                    f"Sample points from frame 2: {pts2[:min(3, len(pts2))].reshape(-1, 2)}"
                )

        return pts1, pts2

    def compute_fundamental_matrix(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        method: int = cv2.FM_RANSAC,
        ransac_threshold: float = 3.0,
        confidence: float = 0.99,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Compute fundamental matrix using matched points.

        Args:
            pts1: Points from first image (N x 1 x 2)
            pts2: Points from second image (N x 1 x 2)
            method: Method for fundamental matrix computation (default: FM_RANSAC)
            ransac_threshold: RANSAC threshold (default: 3.0)
            confidence: RANSAC confidence (default: 0.99)

        Returns:
            Tuple containing:
            - F: Fundamental matrix (3x3) or None if computation failed
            - mask: Inlier mask or None if computation failed
        """
        if len(pts1) < 8 or len(pts2) < 8:
            if self.verbose:
                print(
                    "Warning: Need at least 8 points for fundamental matrix computation"
                )
            return None, None

        # Find fundamental matrix using RANSAC
        F, mask = cv2.findFundamentalMat(
            pts1, pts2, method, ransac_threshold, confidence
        )

        if F is not None and mask is not None:
            inliers = np.sum(mask)
            inlier_ratio = inliers / len(pts1) * 100

            if self.verbose:
                print(f"Fundamental matrix computed successfully")
                print(f"Inliers: {inliers}/{len(pts1)} ({inlier_ratio:.1f}%)")
                print(f"Fundamental matrix:")
                print(F)
        else:
            if self.verbose:
                print("Failed to compute fundamental matrix")

        return F, mask

    def analyze_match_quality(self, good_matches: List[cv2.DMatch]) -> Dict[str, float]:
        """
        Analyze the quality of matches.

        Args:
            good_matches: List of good matches

        Returns:
            Dictionary containing match quality metrics:
            - avg_distance: Average match distance
            - min_distance: Minimum match distance
            - max_distance: Maximum match distance
            - std_distance: Standard deviation of distances
        """
        if len(good_matches) == 0:
            return {
                "avg_distance": float("inf"),
                "min_distance": float("inf"),
                "max_distance": float("inf"),
                "std_distance": 0.0,
            }

        distances = [m.distance for m in good_matches]

        metrics = {
            "avg_distance": np.mean(distances),
            "min_distance": np.min(distances),
            "max_distance": np.max(distances),
            "std_distance": np.std(distances),
        }

        if self.verbose:
            print(f"Match quality analysis:")
            print(f"  Average distance: {metrics['avg_distance']:.2f}")
            print(f"  Min distance: {metrics['min_distance']:.2f}")
            print(f"  Max distance: {metrics['max_distance']:.2f}")
            print(f"  Std deviation: {metrics['std_distance']:.2f}")

        return metrics

    def visualize_matches(
        self,
        img1: np.ndarray,
        keypoints1: List[cv2.KeyPoint],
        img2: np.ndarray,
        keypoints2: List[cv2.KeyPoint],
        good_matches: List[cv2.DMatch],
        max_matches: int = 50,
    ) -> np.ndarray:
        """
        Create visualization of feature matches.

        Args:
            img1: First image
            keypoints1: Keypoints from first image
            img2: Second image
            keypoints2: Keypoints from second image
            good_matches: List of good matches
            max_matches: Maximum number of matches to draw (default: 50)

        Returns:
            Combined image showing matches
        """
        # Draw top matches (limited by max_matches)
        matches_to_draw = good_matches[: min(max_matches, len(good_matches))]

        img_matches = cv2.drawMatches(
            img1,
            keypoints1,
            img2,
            keypoints2,
            matches_to_draw,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        if self.verbose:
            print(f"Visualization created with {len(matches_to_draw)} matches")

        return img_matches

    def visualize_keypoints(
        self, img: np.ndarray, keypoints: List[cv2.KeyPoint]
    ) -> np.ndarray:
        """
        Create visualization of keypoints on an image.

        Args:
            img: Input image
            keypoints: List of keypoints to draw

        Returns:
            Image with keypoints drawn
        """
        img_with_keypoints = cv2.drawKeypoints(
            img,
            keypoints,
            None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )

        if self.verbose:
            print(f"Keypoint visualization created with {len(keypoints)} keypoints")

        return img_with_keypoints

    def set_distance_threshold(self, threshold: float):
        """
        Update the distance threshold for good matches.

        Args:
            threshold: New distance threshold
        """
        self.distance_threshold = threshold
        if self.verbose:
            print(f"Distance threshold updated to: {threshold}")

    def get_match_statistics(
        self, matches: List[cv2.DMatch], good_matches: List[cv2.DMatch]
    ) -> Dict[str, any]:
        """
        Get comprehensive statistics about the matching results.

        Args:
            matches: All matches
            good_matches: Good matches

        Returns:
            Dictionary containing matching statistics
        """
        stats = {
            "total_matches": len(matches),
            "good_matches": len(good_matches),
            "good_match_ratio": (
                len(good_matches) / len(matches) if len(matches) > 0 else 0.0
            ),
            "distance_threshold": self.distance_threshold,
        }

        if len(good_matches) > 0:
            distances = [m.distance for m in good_matches]
            stats.update(
                {
                    "avg_good_distance": np.mean(distances),
                    "min_good_distance": np.min(distances),
                    "max_good_distance": np.max(distances),
                }
            )

        if self.verbose:
            print(f"Matching statistics:")
            print(f"  Total matches: {stats['total_matches']}")
            print(f"  Good matches: {stats['good_matches']}")
            print(f"  Good match ratio: {stats['good_match_ratio']:.1%}")

        return stats


def create_orb_matcher(
    distance_threshold: float = 50.0, cross_check: bool = True, verbose: bool = False
) -> ORBMatcher:
    """
    Factory function to create an ORB matcher with standard settings.

    Args:
        distance_threshold: Maximum Hamming distance for good matches (default: 50.0)
        cross_check: Enable cross-check validation (default: True)
        verbose: Enable verbose output (default: False)

    Returns:
        Configured ORBMatcher instance
    """
    return ORBMatcher(
        distance_threshold=distance_threshold, cross_check=cross_check, verbose=verbose
    )
