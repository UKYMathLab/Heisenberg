# How to run
In `config.ini`, change the values according to the calculation you want to
do. Many of the settings are described in the documentation of the code, but
basically:
- `num_dilates` indicates which dilate you want to
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
- `show_individuals` determines whether or not to show the separate polytopes
  computed. Should be a boolean.
- `show_progression` determines whether or not to show the combined polytopes as
  as a single set of points. Should be a boolean.

After changing the settings, simply navigate to `src/` and run
`dilate.py` from the command line.
