import cv2
import numpy as np
    
class Effect:
    """
    Base class for image/video effects.
    """

    def __init__(self, frame, mask=None):
        """
        Initialize the effect with a frame and optional region and shape.
        """
        self.frame = frame
        self.out = frame.copy()
        self.mask = mask if mask is not None else np.zeros(frame.shape[:2], dtype=np.uint8)

    def apply_mask(self):
        """
        Ensure the effect is only visible where mask == 255. Restore original frame elsewhere.
        """
        self.out[self.mask != 255] = self.frame[self.mask != 255]


class RollPixelsEffect(Effect):
    """
    Apply a pixel roll effect to a video frame.
    Optionally restrict the effect to a region defined by a mask or shape.
    """

    def __init__(self, frame, which="row", shift_length=10, **kwargs):
        super().__init__(frame, **kwargs)
        self.which = which
        self.shift_length = shift_length

    def apply(self):
        """
        Apply the glitch effect to the frame.
        which: 'row' or 'col' (default uses self.which)
        """
        nrows, ncols = self.frame.shape[:2]
        if self.which == "row":
            self._row_shift()
        elif self.which == "col":
            self._col_shift()
        return self.out

    def _row_shift(self):
        for i in np.where(np.any(self.mask == 255, axis=1))[0]:
            shift = self.shift_length
            row_mask = self.mask[i, :]
            if np.any(row_mask == 255):
                for c in range(3):
                    region_pixels = self.out[i, :, c]
                    region_pixels[row_mask == 255] = np.roll(
                        region_pixels[row_mask == 255], shift
                    )

    def _col_shift(self):
        for j in np.where(np.any(self.mask == 255, axis=0))[0]:
            shift = self.shift_length
            col_mask = self.mask[:, j]
            if np.any(col_mask == 255):
                for c in range(3):
                    region_pixels = self.out[:, j, c]
                    region_pixels[col_mask == 255] = np.roll(
                        region_pixels[col_mask == 255], shift
                    )


class RandomRollPixelsEffect(RollPixelsEffect):
    def __init__(
        self,
        frame,
        which="row",
        group_size=1,
        shift_range=(-5, 5),
        **kwargs
    ):
        super().__init__(frame, **kwargs)
        self.which = which
        self.group_size = group_size
        self.shift_range = shift_range

    def apply(self):
        """
        Apply the glitch effect to the frame.
        which: 'row' or 'col' (default uses self.which)
        """
        nrows, ncols = self.frame.shape[:2]
        if self.which == "row":
            self._random_row_shift(nrows)
        elif self.which == "col":
            self._random_col_shift(ncols)
        return self.out

    def _random_row_shift(self, nrows):
        block_starts = [
            i
            for i in range(0, nrows, self.group_size)
            if np.any(self.mask[i : min(i + self.group_size, nrows), :] == 255)
        ]
        for i in block_starts:
            end = min(i + self.group_size, nrows)
            block_mask = self.mask[i:end, :]
            shift = np.random.randint(*self.shift_range)
            for c in range(3):
                region_pixels = self.out[i:end, :, c]
                region_pixels[block_mask == 255] = np.roll(
                    region_pixels[block_mask == 255], shift
                )

    def _random_col_shift(self, ncols):
        block_starts = [
            j
            for j in range(0, ncols, self.group_size)
            if np.any(self.mask[:, j : min(j + self.group_size, ncols)] == 255)
        ]
        for j in block_starts:
            end = min(j + self.group_size, ncols)
            block_mask = self.mask[:, j:end]
            shift = np.random.randint(*self.shift_range)
            for c in range(3):
                region_pixels = self.out[:, j:end, c]
                region_pixels[block_mask == 255] = np.roll(
                    region_pixels[block_mask == 255], shift
                )


class ColorValueEffect(Effect):
    """
    Apply a color value shift effect to a video frame.
    Optionally restrict the effect to a region defined by a mask or shape.
    """

    def __init__(self, frame, shift_value=50, **kwargs):
        super().__init__(frame, **kwargs)
        self.shift_value = shift_value

    def apply(self):
        """
        Apply the color shift effect to the frame.
        """
        self.out = cv2.add(self.frame, self.shift_value)
        self.apply_mask()
        return self.out


class ColorChannelSplitEffect(Effect):
    """
    Apply a color channel split effect to a video frame.
    Optionally restrict the effect to a region defined by a mask or shape.
    """

    def __init__(
        self,
        frame,
        split_distance=10,
        which="row",
        order="bgr",
        **kwargs
    ):
        super().__init__(frame, **kwargs)
        self.split_distance = split_distance
        self.which = which
        self.order = order

    def _shift_color_channel(self, channel, shift):
        if self.which == "row":
            # Iterate over rows with mask coverage
            for i in np.where(np.any(self.mask == 255, axis=1))[0]:
                row_mask = self.mask[i, :]
                # Only roll masked pixels in the row
                row = channel[i, :].copy()
                row[row_mask == 255] = np.roll(row[row_mask == 255], shift)
                channel[i, :] = row
        elif self.which == "col":
            # Iterate over columns with mask coverage
            for j in np.where(np.any(self.mask == 255, axis=0))[0]:
                col_mask = self.mask[:, j]
                col = channel[:, j].copy()
                col[col_mask == 255] = np.roll(col[col_mask == 255], shift)
                channel[:, j] = col

    def apply(self):
        channel_lookup = {"r": 2, "g": 1, "b": 0}  # OpenCV uses BGR
        self.out = self.frame.copy()
        for channel in self.order:
            channel_index = channel_lookup[channel]
            shift = self.split_distance * (self.order.index(channel))
            self._shift_color_channel(self.out[:, :, channel_index], shift)
        return self.out


class CorruptionEffect(Effect):
    """
    Apply a corruption effect to a video frame.
    Optionally restrict the effect to a region defined by a mask or shape.
    """

    def __init__(
        self, frame, corruption_type="random", bitsize=8, **kwargs
    ):
        super().__init__(frame, **kwargs)
        self.corruption_type = corruption_type
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

        self.apply_mask()
        return self.out


class SortEffect(Effect):
    """
    Apply a sorting effect to a video frame.
    Optionally restrict the effect to a region defined by a mask or shape.
    """

    def __init__(
        self,
        frame,
        which="row",
        sort_by="value",
        reverse=False,
        **kwargs
    ):
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
            sort_row_func = lambda i, rm: self.frame[i, rm == 255][
                np.argsort(np.average(self.frame[i, rm == 255], axis=1))
            ]
            sort_col_func = lambda j, rm: self.frame[rm == 255, j][
                np.argsort(np.average(self.frame[rm == 255, j], axis=1))
            ]

        elif self.sort_by in list(lookup.keys()):
            sort_row_func = lambda i, rm: self.frame[i, rm == 255][
                np.argsort(self.frame[i, rm == 255], axis=[lookup[self.sort_by]])
            ]
            sort_col_func = lambda j, rm: self.frame[rm == 255, j][
                np.argsort(self.frame[rm == 255, j])[lookup[self.sort_by]]
            ]

        if self.which == "row":
            for i in np.where(np.any(self.mask == 255, axis=1))[0]:
                row_mask = self.mask[i, :]
                if np.any(row_mask == 255):
                    self.out[i, row_mask == 255] = sort_row_func(i,row_mask)
                    if self.reverse:
                        self.out[i, row_mask == 255] = np.flip(
                            self.out[i, row_mask == 255], axis=0
                        )
        elif self.which == "col":
            for j in np.where(np.any(self.mask == 255, axis=0))[0]:
                col_mask = self.mask[:, j]
                if np.any(col_mask == 255):
                    self.out[col_mask == 255, j] = sort_col_func(j,col_mask)
                    if self.reverse:
                        self.out[col_mask == 255, j] = np.flip(
                            self.out[col_mask == 255, j]
                        )
        return self.out

class ColorPaletteReductionEffect(Effect):
    """
    Reduce the number of colors in a region, simulating posterization or hardware limitations.
    """
    def __init__(self, frame, num_colors=8, **kwargs):
        super().__init__(frame, **kwargs)
        self.num_colors = num_colors

    def apply(self):
        # Quantize each channel to num_colors levels
        levels = np.linspace(0, 255, self.num_colors, dtype=np.uint8)
        quantized = np.zeros_like(self.frame)
        for c in range(self.frame.shape[2]):
            channel = self.frame[:, :, c]
            idx = np.digitize(channel, levels, right=True) - 1
            quantized[:, :, c] = levels[idx]
        self.out = quantized
        self.apply_mask()
        return self.out
