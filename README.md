# Python Video Effects Library

This library provides video effects focused on adding atypical artifacts (glitch, datamosh, analog noise, compression errors, etc.).

## Features
- Modular effect system
- Easy integration with OpenCV and moviepy
- Example scripts and tests

## Getting Started
Install dependencies:
```bash
pip install opencv-python moviepy numpy
```

## Usage
See `video_effects/` for effect modules and `tests/` for usage examples.

### Example: Apply Glitch Effect to a Region
```python
import cv2
import numpy as np
from video_effects.glitch import GlitchEffect

# Load a frame (replace with your image path)
frame = cv2.imread('input.jpg')

# Define a rectangle region (x, y, w, h)
region = (50, 50, 100, 100)
effect = GlitchEffect()
glitched = effect.apply(frame, region=region, shape='rectangle')

# Save or display the result
cv2.imwrite('output.jpg', glitched)
```

## License
MIT
