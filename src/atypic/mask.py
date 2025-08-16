import cv2
import numpy as np

"""
mask.py
This module provides classes and methods for creating and managing masks for image effects.
"""
class SubMasker:
    """
    A class to represent an individual mask element.
    This is used to create and manage individual masks generated in the Masker class.
    """
    def __init__(self, mask):
        """
        Args:
        mask (np.ndarray): The mask array.
        """
        self.mask = mask
            
class Masker:
    """
    A class to create and manage masks for video frames.
    It allows for the creation of various shapes and patterns as masks, which can be applied to video frames.
    The masks can be combined using different behaviors such as 'or', 'and', 'xor', 'nand', and 'nor'.
    """

    def __init__(self, frame, region=None, shape=None, behavior='or'):
        """
        Args:
        frame (np.ndarray): The video frame to apply masks to.
        region (Any, optional): Optional region of interest in the frame (not used in this implementation).
        shape (Any, optional): Optional shape of the mask (not used in this implementation).
        behavior (str, optional): The default behavior for combining masks ('or', 'and', 'xor', 'nand', 'nor').
        """
        self.frame = frame
        self.region = region
        self.shape = shape
        self.behavior = behavior
        self.mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    def _process_behavior(self, mask_element, behavior=None):
        """
        Process the mask element based on the specified behavior.
        Args:
        mask_element (SubMasker): The SubMasker instance containing the mask to be processed.
        behavior (str, optional): The behavior to apply when combining submasks ('or', 'and', 'xor', 'nand', 'nor'). If None, uses the default behavior set during initialization.
        """
        checked_behavior = behavior if behavior else self.behavior
        if  checked_behavior == 'or':
            cv2.bitwise_or(self.mask, mask_element.mask, self.mask)
        elif checked_behavior == 'xor':
            cv2.bitwise_xor(self.mask, mask_element.mask, self.mask)
        elif checked_behavior == 'nand':
            cv2.bitwise_and(self.mask, cv2.bitwise_not(mask_element.mask), self.mask)
        elif checked_behavior == 'and':
            cv2.bitwise_and(self.mask, mask_element.mask, self.mask)
        elif checked_behavior == 'nor':
            cv2.bitwise_not(mask_element.mask, mask_element.mask)
            cv2.bitwise_or(self.mask, mask_element.mask, self.mask)
        else:
            raise ValueError(f"Unknown behavior: {checked_behavior}")
    

    def create_circle_mask(self, center, radius, behavior=None):
        """
        Create a circular mask.
        Args:
        center (tuple[int, int] | tuple[float, float]): (x, y) coordinates of the circle center
        radius (int | float): Radius of the circle
        behavior (str, optional): Mask combination behavior
        """
        mask_element = SubMasker(np.zeros(self.frame.shape[:2], dtype=np.uint8))
        h, w = self.frame.shape[:2]
        if isinstance(radius, float):
            radius = int(min(h, w) * radius)
        if all(isinstance(val, float) for val in center):
            center = (int(w * center[0]), int(h * center[1]))
        cv2.circle(mask_element.mask, center, radius, color=255, thickness=-1)
        self._process_behavior(mask_element, behavior=behavior if behavior else self.behavior)
        return mask_element
    
    def create_rectangle_mask(self, top_left, bottom_right, behavior=None):
        """
        Create a rectangular mask.
        Args:
        top_left (tuple[int, int] | tuple[float, float]): (x, y) coordinates of the top-left corner
        bottom_right (tuple[int, int] | tuple[float, float]): (x, y) coordinates of the bottom-right corner
        behavior (str, optional): Mask combination behavior
        """
        h, w = self.frame.shape[:2]
        # Handle proportional floats
        if any(isinstance(val, float) for val in top_left):
            top_left = (int(w * top_left[0]) if isinstance(top_left[0], float) else top_left[0],
                        int(h * top_left[1]) if isinstance(top_left[1], float) else top_left[1])
        if any(isinstance(val, float) for val in bottom_right):
            bottom_right = (int(w * bottom_right[0]) if isinstance(bottom_right[0], float) else bottom_right[0],
                           int(h * bottom_right[1]) if isinstance(bottom_right[1], float) else bottom_right[1])
        mask_element = SubMasker(np.zeros(self.frame.shape[:2], dtype=np.uint8))
        cv2.rectangle(mask_element.mask, top_left, bottom_right, color=255, thickness=-1)
        self._process_behavior(mask_element, behavior=behavior if behavior else self.behavior)
        return mask_element

    def create_polygon_mask(self, points, behavior=None):
        """
        Create a polygonal mask.
        Args:
        points (list[tuple[int, int] | tuple[float, float]]): List of (x, y) coordinates defining the polygon vertices
        behavior (str, optional): Mask combination behavior
        """
        h, w = self.frame.shape[:2]
        # Handle proportional floats for each point
        pts = []
        for pt in points:
            if all(isinstance(val, float) for val in pt):
                pts.append([int(w * pt[0]), int(h * pt[1])])
            else:
                pts.append([pt[0], pt[1]])
        pts = np.array(pts, np.int32)
        mask_element = SubMasker(np.zeros(self.frame.shape[:2], dtype=np.uint8))
        cv2.fillPoly(mask_element.mask, [pts], color=255)
        self._process_behavior(mask_element,behavior=behavior if behavior else self.behavior)
        return mask_element
    def create_ellipse_mask(self, center, axes, angle=0, behavior=None):
        """
        Create an elliptical mask.
        Args:
        center (tuple[int, int] | tuple[float, float]): (x, y) coordinates of the ellipse center
        axes (tuple[int, int] | tuple[float, float]): (major_axis_length, minor_axis_length)
        angle (float, optional): Rotation angle of the ellipse in degrees
        behavior (str, optional): Mask combination behavior
        """
        h, w = self.frame.shape[:2]
        # Handle proportional floats
        if all(isinstance(val, float) for val in center):
            center = (int(w * center[0]), int(h * center[1]))
        if all(isinstance(val, float) for val in axes):
            axes = (int(w * axes[0]), int(h * axes[1]))
        mask_element = SubMasker(np.zeros(self.frame.shape[:2], dtype=np.uint8))
        cv2.ellipse(mask_element.mask, center, axes, angle, 0, 360, color=255, thickness=-1)
        self._process_behavior(mask_element,behavior=behavior if behavior else self.behavior)
        return mask_element


    def create_band_mask(self, orientation="horizontal", start=0, end=None, behavior=None):
        """
        Create a horizontal or vertical band mask.
        Args:
        orientation (str): 'horizontal' or 'vertical'
        start (int | float): Starting index of the band
        end (int | float, optional): Ending index of the band (defaults to frame edge)
        behavior (str, optional): Mask combination behavior
        """
        h, w = self.frame.shape[:2]
        # Handle proportional floats
        if isinstance(start, float):
            start = int((h if orientation=="horizontal" else w) * start)
        if isinstance(end, float):
            end = int((h if orientation=="horizontal" else w) * end)
        mask_element = SubMasker(np.zeros(self.frame.shape[:2], dtype=np.uint8))
        if orientation == "horizontal":
            if end is None:
                end = h
            mask_element.mask[start:end, :] = 255
        elif orientation == "vertical":
            if end is None:
                end = w
            mask_element.mask[:, start:end] = 255
        self._process_behavior(mask_element,behavior=behavior if behavior else self.behavior)
        return mask_element


    def create_checkerboard_mask(self, block_size=10, behavior=None):
        """
        Create a checkerboard pattern mask.
        Args:
        block_size (int | float): Size of each square block
        behavior (str, optional): Mask combination behavior
        """
        h, w = self.frame.shape[:2]
        # Handle proportional float for block_size
        if isinstance(block_size, float):
            block_size = int(min(h, w) * block_size)
        mask_element = SubMasker(np.zeros(self.frame.shape[:2], dtype=np.uint8))
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                if (i // block_size + j // block_size) % 2 == 0:
                    mask_element.mask[i:i+block_size, j:j+block_size] = 255
        self._process_behavior(mask_element,behavior=behavior if behavior else self.behavior)
        return mask_element


    def create_stripe_mask(self, orientation="horizontal", stripe_width=10, gap=10, behavior=None):
        """
        Create a striped mask.
        Args:
        orientation (str): 'horizontal' or 'vertical'
        stripe_width (int | float): Width of each stripe
        gap (int | float): Gap between stripes
        behavior (str, optional): Mask combination behavior
        """
        h, w = self.frame.shape[:2]
        # Handle proportional floats
        if isinstance(stripe_width, float):
            stripe_width = int((h if orientation=="horizontal" else w) * stripe_width)
        if isinstance(gap, float):
            gap = int((h if orientation=="horizontal" else w) * gap)
        mask_element = SubMasker(np.zeros(self.frame.shape[:2], dtype=np.uint8))
        if orientation == "horizontal":
            for i in range(0, h, stripe_width + gap):
                mask_element.mask[i:i+stripe_width, :] = 255
        elif orientation == "vertical":
            for j in range(0, w, stripe_width + gap):
                mask_element.mask[:, j:j+stripe_width] = 255
        self._process_behavior(mask_element,behavior=behavior if behavior else self.behavior)
        return mask_element
    def create_full_mask(self, behavior=None):
        """
        Create a mask that covers the entire frame.
        Args:
        behavior (str, optional): Mask combination behavior
        """
        mask_element = SubMasker(np.ones(self.frame.shape[:2], dtype=np.uint8)*255)
        self._process_behavior(mask_element,behavior=behavior if behavior else self.behavior)
        return mask_element
