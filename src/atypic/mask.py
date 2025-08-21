import cv2
import numpy as np

"""
This module provides classes and methods for creating and managing masks for image effects.
"""


class Masker:
    """
    A class to create and manage masks for image frames.
    
    It allows for the creation of various shapes and patterns as masks, which can be applied to image frames.
    The masks can be combined using different behaviors such as 'or', 'and', 'xor', 'nand', and 'nor'.
    """

    def __init__(self, frame, behavior="add"):
        """
        Args:
            frame (np.ndarray): The image frame to apply masks to.
            behavior (str, optional): The default behavior for combining masks ('or', 'and', 'xor', 'nand', 'nor').
        """
        self.frame = frame
        self.behavior = behavior
        self.mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    def _process_behavior(self, mask_element, behavior=None, opacity=255):
        """
        Process the mask element based on the specified behavior.
        
        Args:
            mask_element (Masker): The Masker instance containing the mask to be processed.
            behavior (str, optional): The behavior to apply when combining submasks ('or', 'and', 'xor', 'nand', 'nor'). If None, uses the default behavior set during initialization.
        """
        mask_element.mask[mask_element.mask==255] = opacity
        checked_behavior = behavior if behavior else self.behavior
        if checked_behavior == "add":
            cv2.add(self.mask, mask_element.mask, self.mask)
        elif checked_behavior == "subtract":
            cv2.subtract(self.mask, mask_element.mask, self.mask)
        elif checked_behavior == "xor":
            cv2.bitwise_xor(self.mask, mask_element.mask, self.mask)
        elif checked_behavior == "nand":
            cv2.bitwise_and(self.mask, cv2.bitwise_not(mask_element.mask), self.mask)
        elif checked_behavior == "and":
            cv2.bitwise_and(self.mask, mask_element.mask, self.mask)
        elif checked_behavior == "nor":
            cv2.bitwise_not(mask_element.mask, mask_element.mask)
            cv2.bitwise_or(self.mask, mask_element.mask, self.mask)
        else:
            raise ValueError(f"Unknown behavior: {checked_behavior}")

    def create_band(self, orientation="horizontal", start=0, end=None, behavior=None, opacity=255):
        """
        Create a horizontal or vertical band mask.
        
        Args:
            orientation (str): 'horizontal' or 'vertical'
            start (int | float): Starting index of the band
            end (int | float, optional): Ending index of the band (defaults to frame edge)
            behavior (str, optional): Masker combination behavior
            
        Returns:
            Masker: A Masker instance with the band mask.
        """
        h, w = self.frame.shape[:2]
        # Handle proportional floats
        if isinstance(start, float):
            start = int((h if orientation == "horizontal" else w) * start)
        if isinstance(end, float):
            end = int((h if orientation == "horizontal" else w) * end)
        mask_element = Masker(np.zeros(self.frame.shape[:2], dtype=np.uint8))
        if orientation == "horizontal":
            if end is None:
                end = h
            mask_element.mask[start:end, :] = 255
        elif orientation == "vertical":
            if end is None:
                end = w
            mask_element.mask[:, start:end] = 255
        self._process_behavior(
            mask_element, behavior=behavior if behavior else self.behavior, opacity=opacity
        )
        return mask_element

    def create_checkerboard(self, block_size=10, behavior=None, opacity=255):
        """
        Create a checkerboard pattern mask.
        
        Args:
            block_size (int | float): Size of each square block
            behavior (str, optional): Masker combination behavior
            
        Returns:
            Masker: A Masker instance with the checkerboard mask.
        """
        h, w = self.frame.shape[:2]
        # Handle proportional float for block_size
        if isinstance(block_size, float):
            block_size = int(min(h, w) * block_size)
        mask_element = Masker(np.zeros(self.frame.shape[:2], dtype=np.uint8))
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                if (i // block_size + j // block_size) % 2 == 0:
                    mask_element.mask[i : i + block_size, j : j + block_size] = 255
        self._process_behavior(
            mask_element, behavior=behavior if behavior else self.behavior, opacity=opacity
        )
        return mask_element

    def create_circle(self, center, radius, behavior=None, opacity=255):
        """
        Create a circular mask.
        
        Args:
            center (tuple[int, int] | tuple[float, float]): (x, y) coordinates of the circle center
            radius (int | float): Radius of the circle
            behavior (str, optional): Masker combination behavior
            
        Returns:
            Masker: A Masker instance with the circular mask.
        """
        mask_element = Masker(np.zeros(self.frame.shape[:2], dtype=np.uint8))
        h, w = self.frame.shape[:2]
        if isinstance(radius, float):
            radius = int(min(h, w) * radius)
        if all(isinstance(val, float) for val in center):
            center = (int(w * center[0]), int(h * center[1]))
        cv2.circle(mask_element.mask, center, radius, color=255, thickness=-1)
        self._process_behavior(
            mask_element, behavior=behavior if behavior else self.behavior, opacity=opacity
        )
        return mask_element

    def create_ellipse(self, center, axes, angle=0, behavior=None, opacity=255):
        """
        Create an elliptical mask.
        
        Args:
            center (tuple[int, int] | tuple[float, float]): (x, y) coordinates of the ellipse center
            axes (tuple[int, int] | tuple[float, float]): (major_axis_length, minor_axis_length)
            angle (float, optional): Rotation angle of the ellipse in degrees
            behavior (str, optional): Masker combination behavior
            
        Returns:
            Masker: A Masker instance with the elliptical mask.
        """
        h, w = self.frame.shape[:2]
        # Handle proportional floats
        if all(isinstance(val, float) for val in center):
            center = (int(w * center[0]), int(h * center[1]))
        if all(isinstance(val, float) for val in axes):
            axes = (int(w * axes[0]), int(h * axes[1]))
        mask_element = Masker(np.zeros(self.frame.shape[:2], dtype=np.uint8))
        cv2.ellipse(
            mask_element.mask, center, axes, angle, 0, 360, color=255, thickness=-1
        )
        self._process_behavior(
            mask_element, behavior=behavior if behavior else self.behavior, opacity=opacity
        )
        return mask_element

    def create_polygon(self, points, behavior=None, opacity=255):
        """
        Create a polygonal mask.
        
        Args:
            points (list[tuple[int, int] | tuple[float, float]]): List of (x, y) coordinates defining the polygon vertices
            behavior (str, optional): Masker combination behavior
            
        Returns:
            Masker: A Masker instance with the polygon mask.
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
        mask_element = Masker(np.zeros(self.frame.shape[:2], dtype=np.uint8))
        cv2.fillPoly(mask_element.mask, [pts], color=255)
        self._process_behavior(
            mask_element, behavior=behavior if behavior else self.behavior, opacity=opacity
        )
        return mask_element

    def create_rectangle(self, top_left, bottom_right, behavior=None, opacity=255):
        """
        Create a rectangular mask.
        
        Args:
            top_left (tuple[int, int] | tuple[float, float]): (x, y) coordinates of the top-left corner
            bottom_right (tuple[int, int] | tuple[float, float]): (x, y) coordinates of the bottom-right corner
            behavior (str, optional): Masker combination behavior
            
        Returns:
            Masker: A Masker instance with the rectangular mask.
        """
        h, w = self.frame.shape[:2]
        # Handle proportional floats
        if any(isinstance(val, float) for val in top_left):
            top_left = (
                int(w * top_left[0]) if isinstance(top_left[0], float) else top_left[0],
                int(h * top_left[1]) if isinstance(top_left[1], float) else top_left[1],
            )
        if any(isinstance(val, float) for val in bottom_right):
            bottom_right = (
                (
                    int(w * bottom_right[0])
                    if isinstance(bottom_right[0], float)
                    else bottom_right[0]
                ),
                (
                    int(h * bottom_right[1])
                    if isinstance(bottom_right[1], float)
                    else bottom_right[1]
                ),
            )
        mask_element = Masker(np.zeros(self.frame.shape[:2], dtype=np.uint8))
        cv2.rectangle(
            mask_element.mask, top_left, bottom_right, color=255, thickness=-1
        )
        self._process_behavior(
            mask_element, behavior=behavior if behavior else self.behavior, opacity=opacity
        )
        return mask_element

    def create_regular_polygon(self, center, radius, sides=3, behavior=None, opacity=255):
        """
        Create a regular polygon mask.
        
        Args:
            center (tuple[int, int] | tuple[float, float]): (x, y) coordinates of the polygon center
            radius (int | float): Radius of the circumscribed circle
            sides (int): Number of sides of the polygon
            behavior (str, optional): Masker combination behavior
            
        Returns:
            Masker: A Masker instance with the regular polygon mask.
        """
        h, w = self.frame.shape[:2]
        # Handle proportional floats
        if isinstance(radius, float):
            radius = int(min(h, w) * radius)
        if all(isinstance(val, float) for val in center):
            center = (int(w * center[0]), int(h * center[1]))
        if sides < 3:
            raise ValueError("Polygon must have at least 3 sides")
        mask_element = Masker(np.zeros(self.frame.shape[:2], dtype=np.uint8))
        angle = 2 * np.pi / sides
        points = []
        for i in range(sides):
            theta = i * angle
            x = int(center[0] + radius * np.cos(theta))
            y = int(center[1] + radius * np.sin(theta))
            points.append((x, y))
        self.create_polygon_mask(
            points, behavior=behavior if behavior else self.behavior
        )

        self._process_behavior(
            mask_element, behavior=behavior if behavior else self.behavior, opacity=opacity
        )
        return mask_element

    def create_star(self, center, radius, points=5, behavior=None, opacity=255):
        """
        Create a star-shaped mask.
        
        Args:
            center (tuple[int, int] | tuple[float, float]): (x, y) coordinates of the star center
            radius (int | float): Radius of the star points
            points (int): Number of points in the star
            behavior (str, optional): Masker combination behavior
            
        Returns:
            Masker: A Masker instance with the star-shaped mask.
        """
        h, w = self.frame.shape[:2]
        # Handle proportional floats
        if isinstance(radius, float):
            radius = int(min(h, w) * radius)
        if all(isinstance(val, float) for val in center):
            center = (int(w * center[0]), int(h * center[1]))
        if points < 3:
            raise ValueError("Star must have at least 3 points")
        mask_element = Masker(np.zeros(self.frame.shape[:2], dtype=np.uint8))
        angle = 2 * np.pi / points
        point_coords = []
        for i in range(points):
            theta = i * angle
            x1 = int(center[0] + radius * np.cos(theta))
            y1 = int(center[1] + radius * np.sin(theta))
            x2 = int(center[0] + radius * np.cos(theta + angle / 2) / 2)
            y2 = int(center[1] + radius * np.sin(theta + angle / 2) / 2)
            point_coords.append((x1, y1))
            point_coords.append((x2, y2))
        mask_element.create_polygon(
            point_coords, behavior=behavior if behavior else self.behavior
        )

        self._process_behavior(
            mask_element, behavior=behavior if behavior else self.behavior, opacity=opacity
        )
        return mask_element

    def create_stripes(
        self, orientation="horizontal", stripe_width=10, gap=10, behavior=None, opacity=255
    ):
        """
        Create a striped mask.
        
        Args:
            orientation (str): 'horizontal' or 'vertical'
            stripe_width (int | float): Width of each stripe
            gap (int | float): Gap between stripes
            behavior (str, optional): Masker combination behavior
            
        Returns:
            Masker: A Masker instance with the striped mask.
        """
        h, w = self.frame.shape[:2]
        # Handle proportional floats
        if isinstance(stripe_width, float):
            stripe_width = int((h if orientation == "horizontal" else w) * stripe_width)
        if isinstance(gap, float):
            gap = int((h if orientation == "horizontal" else w) * gap)
        mask_element = Masker(np.zeros(self.frame.shape[:2], dtype=np.uint8))
        if orientation == "horizontal":
            for i in range(0, h, stripe_width + gap):
                mask_element.mask[i : i + stripe_width, :] = 255
        elif orientation == "vertical":
            for j in range(0, w, stripe_width + gap):
                mask_element.mask[:, j : j + stripe_width] = 255
        self._process_behavior(
            mask_element, behavior=behavior if behavior else self.behavior, opacity=opacity
        )
        return mask_element

    def create_whole_frame(self, behavior=None, opacity=255):
        """
        Create a mask that covers the entire frame.
        
        Args:
            behavior (str, optional): Masker combination behavior
            
        Returns:
            Masker: A Masker instance with a mask covering the entire frame.
        """
        mask_element = Masker(np.ones(self.frame.shape[:2], dtype=np.uint8) * 255)
        self._process_behavior(
            mask_element, behavior=behavior if behavior else self.behavior, opacity=opacity
        )
        return mask_element

    def rotate(self, angle):
        """
        Rotate the instance mask element by a specified angle.
        
        Args:
            angle (float): The angle in degrees to rotate the mask.
        """
        h, w = self.frame.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_mask = cv2.warpAffine(
            self.mask, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR
        )
        self.mask = rotated_mask

    def mirror(self, axis='vertical'):
        """
        Mirror a given mask element in the specified direction.

        Parameters:
            mask_element (Masker): The Masker instance containing the mask to be mirrored.
            axis (str): 'horizontal' or 'vertical'
        """
        if axis == "horizontal":
            mirrored_mask = cv2.flip(self.mask, 1)
        elif axis == "vertical":
            mirrored_mask = cv2.flip(self.mask, 0)
        else:
            raise ValueError("Direction must be 'horizontal' or 'vertical'")
        self.mask = mirrored_mask
        
    def add(self, *mask_elements, behavior=None):
        """
        Combine multiple mask elements into a single mask and add them to current Masker instance.
        
        Args:
            *mask_elements (Masker): The Masker instances to combine.
            behavior (str, optional): Masker combination behavior
            
        Returns:
            Masker: A new Masker instance containing the combined mask.
        """
        combined_mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)
        for mask_element in mask_elements:
            self._process_behavior(mask_element, behavior=behavior)
            combined_mask = cv2.bitwise_or(combined_mask, mask_element.mask)
        mask_element = Masker(combined_mask)
        return mask_element

    def invert(self):
        """
        Invert the current mask of the Masker instance.
        """
        self.mask = cv2.invert(self.mask)