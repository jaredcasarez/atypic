import cv2
import numpy as np

"""
Module for image/video effects.

This module provides various effects that can be applied to image frames, such as color channel splitting,
color palette reduction, color value shifting, pixel corruption, pixel rolling, and sorting.
Each effect is implemented as a subclass of the `Effect` base class, which handles the initialization
and application of the effect to a given frame.
"""


class Effect:
    """Base class for image/video effects.
    """

    def __init__(self, frame, opacity=255, mask=None):
        """
    
        Args:
            frame (numpy.ndarray): The video frame to apply the effect on.
            opacity (int): Opacity of the effect (0-255). 255 is fully visible, 0 is invisible.
            mask (numpy.ndarray, optional): A mask (0-255) defining the region and strength to apply the effect.
                If None, the effect applies to the entire frame.
                
        """
        self.frame = frame
        self.out = frame.copy()
        self.mask = (
            mask if mask is not None else np.zeros(frame.shape[:2], dtype=np.uint8)+255
        )
        self.opacity = opacity

    def blend(self):
        """Blend the original frame and the effected frame using mask and opacity.
        
        Returns:
            numpy.ndarray: The blended frame.
            
        """
        # Compute per-pixel alpha: mask (0-255) * opacity (0-255) / 255
        alpha = (self.mask.astype(np.float32) * self.opacity / 255.0) / 255.0
        # Expand alpha to shape (H, W, 1) for broadcasting
        alpha = np.expand_dims(alpha, axis=-1)
        # Blend: out = frame * (1 - alpha) + effected * alpha
        print(alpha[len(alpha) // 2, len(alpha[0]) // 2])
        print(self.frame[len(self.frame) // 2, len(self.frame[0]) // 2])
        print(self.out[len(self.out) // 2, len(self.out[0]) // 2])
        blended = (
            self.frame.astype(np.float32) * (1 - alpha)
            + self.out.astype(np.float32) * alpha
        )
        print(self.frame[len(self.frame) // 2, len(self.frame[0]) // 2])
        print(blended[len(blended) // 2, len(blended[0]) // 2])
        return blended.astype(np.uint8)


class ColorChannelSplit(Effect):
    """
    Apply a color channel split effect to an image frame.
    
    This effect shifts each color channel (B, G, R) by a specified distance.
    """

    def __init__(self, frame, split_distance=0.1, which="row", order="bgr", **kwargs):
        """
        Initialize the effect with a frame, split distance, direction, and channel order.
        
        Args:
            frame (numpy.ndarray): The video frame to apply the effect on.
            split_distance (int | float): Distance to shift each color channel. If float, it is interpreted as a proportion of the frame size.
            which (str): Direction of the split, either 'row' or 'col'.
            order (str): Order of color channels to apply the effect on ('bgr', 'rgb', etc.).
            **kwargs: Additional keyword arguments for the Effect base class.
        """

        super().__init__(frame, **kwargs)
        if isinstance(split_distance, float):
            split_distance = int(min(frame.shape[:2]) * split_distance)
        self.split_distance = split_distance
        self.which = which
        self.order = order

    def _shift_color_channel(self, channel, shift):
        """
        Shift a single color channel by the specified distance.
        
        Args:
            channel (numpy.ndarray): The color channel to shift.
            shift (int): The distance to shift the channel.
        """

        if self.which == "row":
            # Iterate over rows with mask coverage
            for i in np.where(np.any(self.mask != 0, axis=1))[0]:
                row_mask = self.mask[i, :]
                # Only roll masked pixels in the row
                row = channel[i, :].copy()
                row[row_mask != 0] = np.roll(row[row_mask != 0], shift)
                channel[i, :] = row
        elif self.which == "col":
            # Iterate over columns with mask coverage
            for j in np.where(np.any(self.mask != 0, axis=0))[0]:
                col_mask = self.mask[:, j]
                col = channel[:, j].copy()
                col[col_mask != 0] = np.roll(col[col_mask != 0], shift)
                channel[:, j] = col

    def apply(self):
        """
        Apply the color channel split effect to the frame.
        """
        channel_lookup = {"r": 2, "g": 1, "b": 0}  # OpenCV uses BGR
        self.out = self.frame.copy()
        for channel in self.order:
            channel_index = channel_lookup[channel]
            shift = self.split_distance * (self.order.index(channel))
            self._shift_color_channel(self.out[:, :, channel_index], shift)
        return self.blend()


class ColorPaletteReduction(Effect):
    """
    Apply a color palette reduction effect to an image frame.
    
    This effect reduces the number of colors in the frame to a specified number.
    """

    def __init__(self, frame, num_colors=4, **kwargs):
        """
        Initialize the effect with a frame and number of colors.
        
        Args:
            frame (numpy.ndarray): The video frame to apply the effect on.
            num_colors (int): Number of colors to reduce the frame to.
            **kwargs: Additional keyword arguments for the Effect base class.
        """
        super().__init__(frame, **kwargs)
        self.num_colors = num_colors

    def apply(self):
        """
        Apply the color palette reduction effect to the frame.
        
        This method quantizes each color channel to the specified number of colors.
        """
        levels = np.linspace(0, 255, self.num_colors, dtype=np.uint8)
        quantized = np.zeros_like(self.frame)
        for c in range(self.frame.shape[2]):
            channel = self.frame[:, :, c]
            idx = np.digitize(channel, levels, right=True) - 1
            quantized[:, :, c] = levels[idx]
        self.out = quantized
        return self.blend()


class ColorValue(Effect):
    """
    Apply a color value shift effect to an image frame.
    
    This effect shifts the color values of the frame by a specified amount.
    """

    def __init__(
        self,
        frame,
        shift_values: (
            tuple[int, int, int] | tuple[float, float, float]
        ) = (0.2, 0.2, 0.2),
        **kwargs,
    ):
        """
        Initialize the effect with a frame and shift value.
        
        Args:
            frame (numpy.ndarray): The video frame to apply the effect on.
            shift_value (tuple[float, float, float]): Amount to multiply the color values, for each channel (B, G, R).
            **kwargs: Additional keyword arguments for the Effect base class.
        """
        super().__init__(frame, **kwargs)
        self.shift_values = shift_values

    def apply(self):
        """
        Apply the color shift effect to the frame.
        """
        try:
            self.out = cv2.multiply(self.frame, np.array(self.shift_values, dtype=np.float32))
        except:
            raise ValueError(
                "shift_values must either be convertible to float or all int types."
            )
        return self.blend()


class Corruption(Effect):
    """
    Apply a corruption effect to an image frame.
    
    This effect randomly corrupts blocks of pixels in the frame.
    """

    def __init__(self, frame, corruption_type="random", bitsize=0.01, **kwargs):
        """
        Initialize the effect with a frame, corruption type, and bitsize.
        
        Args:
            frame (numpy.ndarray): The video frame to apply the effect on.
            corruption_type (str): Type of corruption to apply (e.g., 'random').
            bitsize (int | float): Size of the corruption block. If float, it is
                interpreted as a proportion of the frame size.
            **kwargs: Additional keyword arguments for the Effect base class.
        """
        super().__init__(frame, **kwargs)
        self.corruption_type = corruption_type
        if isinstance(bitsize, float):
            bitsize = int(min(frame.shape[:2]) * bitsize)
        self.bitsize = bitsize

    def apply(self):
        """
        Apply the corruption effect to the frame.
        """
        if self.corruption_type == "random":
            for row in range(0, self.frame.shape[0], self.bitsize):
                for col in range(0, self.frame.shape[1], self.bitsize):
                    self.out[row : row + self.bitsize, col : col + self.bitsize, :] = (
                        np.random.randint(0, 256, 1, dtype=np.uint8)
                    )
        else:
            raise ValueError(f"Unknown corruption type: {self.corruption_type}")
        return self.blend()


class RollPixels(Effect):
    """
    Apply a pixel roll effect to an image frame.
    """

    def __init__(self, frame, which="row", shift_length=0.1, **kwargs):
        """
        Initialize the effect with a frame, direction, and shift length.
        
        Args:
            frame (numpy.ndarray): The video frame to apply the effect on.
                which (str): Direction of the roll, either 'row' or 'col'.
            shift_length (int | float): Length of the pixel shift. If float, it is interpreted as a proportion of the frame size.
            **kwargs: Additional keyword arguments for the Effect base class.
        """
        super().__init__(frame, **kwargs)
        self.which = which
        if isinstance(shift_length, float):
            shift_length = int(min(frame.shape[:2]) * shift_length)
        self.shift_length = shift_length

    def apply(self):
        """
        Apply the pixel roll effect to the frame.
        """
        if self.which == "row":
            self._row_shift()
        elif self.which == "col":
            self._col_shift()
        return self.blend()

    def _row_shift(self):
        """
        Shift pixels in each row by the specified length.
        """
        for i in np.where(np.any(self.mask != 0, axis=1))[0]:
            shift = self.shift_length
            row_mask = self.mask[i, :]
            if np.any(row_mask != 0):
                for c in range(3):
                    region_pixels = self.out[i, :, c]
                    region_pixels[row_mask != 0] = np.roll(
                        region_pixels[row_mask != 0], shift
                    )

    def _col_shift(self):
        """
        Shift pixels in each column by the specified length.
        """
        for j in np.where(np.any(self.mask != 0, axis=0))[0]:
            shift = self.shift_length
            col_mask = self.mask[:, j]
            if np.any(col_mask != 0):
                for c in range(3):
                    region_pixels = self.out[:, j, c]
                    region_pixels[col_mask != 0] = np.roll(
                        region_pixels[col_mask != 0], shift
                    )


class RollPixelsRandom(RollPixels):
    """
    Apply a random pixel roll effect to an image frame.
    
    This effect randomly shifts blocks of pixels in either rows or columns.
    """

    def __init__(
        self, frame, which="row", group_size=0.2, shift_range=(0.1, 0.1), **kwargs
    ):
        """
        Initialize the effect with a frame, direction, group size, and shift range.
        
        Args:
            frame (numpy.ndarray): The video frame to apply the effect on.
            which (str): Direction of the roll, either 'row' or 'col'.
            group_size (int | float): Size of the pixel block to shift. If float, it is interpreted as a proportion of the frame size.
            shift_range (tuple[int | float, int | float]): Range of random shifts. If floats, they are interpreted as proportions of the frame size.
            **kwargs: Additional keyword arguments for the Effect base class.
        """
        super().__init__(frame, **kwargs)
        self.which = which
        if isinstance(group_size, float):
            group_size = int(min(frame.shape[:2]) * group_size)
        self.group_size = group_size
        if any(isinstance(end, float) for end in shift_range):
            print(frame.shape[:2])
            shift_range = (
                -int(min(frame.shape[:2]) * shift_range[0]),
                int(min(frame.shape[:2]) * shift_range[1]),
            )
        self.shift_range = shift_range

    def apply(self):
        """Apply the random pixel roll effect to the frame.
        """
        nrows, ncols = self.frame.shape[:2]
        if self.which == "row":
            self._random_row_shift(nrows)
        elif self.which == "col":
            self._random_col_shift(ncols)
        return self.blend()

    def _random_row_shift(self, nrows):
        """
        Shift blocks of pixels in each row by a random amount within the specified range.
        """
        block_starts = [
            i
            for i in range(0, nrows, self.group_size)
            if np.any(self.mask[i : min(i + self.group_size, nrows), :] != 0)
        ]
        for i in block_starts:
            end = min(i + self.group_size, nrows)
            block_mask = self.mask[i:end, :]
            shift = np.random.randint(*self.shift_range)
            for c in range(3):
                region_pixels = self.out[i:end, :, c]
                region_pixels[block_mask != 0] = np.roll(
                    region_pixels[block_mask != 0], shift
                )

    def _random_col_shift(self, ncols):
        """
        Shift blocks of pixels in each column by a random amount within the specified range.
        """
        block_starts = [
            j
            for j in range(0, ncols, self.group_size)
            if np.any(self.mask[:, j : min(j + self.group_size, ncols)] != 0)
        ]
        for j in block_starts:
            end = min(j + self.group_size, ncols)
            block_mask = self.mask[:, j:end]
            shift = np.random.randint(*self.shift_range)
            for c in range(3):
                region_pixels = self.out[:, j:end, c]
                region_pixels[block_mask != 0] = np.roll(
                    region_pixels[block_mask != 0], shift
                )


class Sort(Effect):
    """
    Apply a sorting effect to an image frame.
    
    This effect sorts pixels in a specified direction (row or column) based on their color values.
    """

    def __init__(self, frame, which="row", sort_by="value", reverse=False, **kwargs):
        """
        Initialize the effect with a frame, direction, sorting criteria, and reverse flag.

        Args:
            frame (numpy.ndarray): The video frame to apply the effect on.
                which (str): Direction of the sort, either 'row' or 'col'.
            sort_by (str): Criteria to sort by ('value', 'r', 'g',
                'b'). If 'value', sorts by average color value; if 'r', 'g', or 'b', sorts by that channel.
            reverse (bool): If True, reverses the sort order.
            **kwargs: Additional keyword arguments for the Effect base class.
        """
        super().__init__(frame, **kwargs)
        self.which = which
        self.sort_by = sort_by
        self.reverse = reverse

    def apply(self):
        """
        Apply the sorting effect to the frame.
        """
        lookup = {"r": 2, "g": 1, "b": 0}  # OpenCV uses BGR
        if self.sort_by == "value":
            sort_row_func = lambda i, rm: self.frame[i, rm != 0][
                np.argsort(np.average(self.frame[i, rm != 0], axis=1))
            ]
            sort_col_func = lambda j, rm: self.frame[rm != 0, j][
                np.argsort(np.average(self.frame[rm != 0, j], axis=1))
            ]
        elif self.sort_by in list(lookup.keys()):
            sort_row_func = lambda i, rm: self.frame[i, rm != 0][
                np.argsort(self.frame[i, rm != 0][:, lookup[self.sort_by]])
            ]
            sort_col_func = lambda j, rm: self.frame[rm != 0, j][
                np.argsort(self.frame[rm != 0, j][:, lookup[self.sort_by]])
            ]
        if self.which == "row":
            for i in np.where(np.any(self.mask != 0, axis=1))[0]:
                row_mask = self.mask[i, :]
                if np.any(row_mask != 0):
                    self.out[i, row_mask != 0] = sort_row_func(i, row_mask)
                    if self.reverse:
                        self.out[i, row_mask != 0] = np.flip(
                            self.out[i, row_mask != 0], axis=0
                        )
        elif self.which == "col":
            for j in np.where(np.any(self.mask != 0, axis=0))[0]:
                col_mask = self.mask[:, j]
                if np.any(col_mask != 0):
                    self.out[col_mask != 0, j] = sort_col_func(j, col_mask)
                    if self.reverse:
                        self.out[col_mask != 0, j] = np.flip(self.out[col_mask != 0, j])
        return self.blend()
