## Masker
Masker instances stores the generated mask. Multiple SubMasks can be added with the built-in Masker functions (rectangle, checkerboard, etc.). 

Built-in mask functions (seen below):
* Entire frame
* Circle
* Ellipse
* Rectangle
* Polygon
* Band
* Stripes
* Checkerboard

Mask functions accept accept both pixels (indices) and proportions of the input image (floats) for coordinates/sizes.

## Effect
Effect instances can be fed a Masker's mask (Masker.mask) to constrict its area of effect.
