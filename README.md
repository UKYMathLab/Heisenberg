# How to run
In `config.ini`, change the values according to the calculation you want to
do. Many of the settings are described in the documentation of the code, but
basically:
- `num_dilates` It indicates which dilate you want to
  calculate. Larger dilates take longer to compute; a reasonable maximum value is
  usually `10`. Should be an integer.
- `mode` indicates what operation to calculate with.
    - `n` - normal addition.
    - `h` - Heisenberg group operation.
    - `d` - Duchin's reformulation of the Heisenberg operation.
- `basis` chooses the set of starting points. These are set in
  `src/EfficientHeisenberg.py`.
  - `s` The standard right, unit tetrahedron (unit vectors and the origin).
  - `w` Random (integral) points.
  - `r` The same as option `s` except more points are interpolated between points.

After changing the settings, simply navigate to `src/` and run
`EfficientHeisenberg.py` from the command line.
