import cv2
import numpy as np
from typing import Tuple, List, Optional


class ORBExtractor:
    def __init__(self, n_features: int = 500, scale_factor: float = 1.2, n_levels: int = 8):
        """
        ORB feature extractor for SLAM applications.
        
        Args:
            n_features: Maximum number of features to extract
            scale_factor: Pyramid decimation ratio for ORB
            n_levels: Number of pyramid levels
        """
        self.n_features = n_features
        self.scale_factor = scale_factor
        self.n_levels = n_levels
        
        # Conservative ORB parameters for stable feature detection
        self.orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            edgeThreshold=31,     # Default OpenCV value for robust edge handling
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20      # Default OpenCV value for stable corner detection
        )
    
    def extract_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Extract ORB features from input image.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if descriptors is None:
            descriptors = np.array([])
            
        return keypoints, descriptors
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray, 
                      distance_threshold: int = 50) -> List[cv2.DMatch]:
        """
        Match features between two descriptor sets using cross-check and distance filtering.
        
        Args:
            desc1: First set of descriptors
            desc2: Second set of descriptors
            distance_threshold: Maximum Hamming distance for good matches
            
        Returns:
            List of good matches
        """
        if desc1.size == 0 or desc2.size == 0:
            return []
            
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        
        # Filter matches by distance threshold
        good_matches = [m for m in matches if m.distance < distance_threshold]
        
        # Sort by distance (best matches first)
        good_matches = sorted(good_matches, key=lambda x: x.distance)
                    
        return good_matches