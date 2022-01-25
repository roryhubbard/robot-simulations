import numpy as np
from math import factorial
import cvxpy as cp




class PiecewisePolynomial:

  def __init__(self, ts, nflats, poly_degree, smoothness_degree):
    self.ts = ts
    self.ns = self.ts.size - 1 # number of splines
    self.nflats = nflats
    self.poly_degree = poly_degree
    self.smoothness_degree = smoothness_degree
    self.spline_coeffs = [
      cp.Variable((nflats, poly_degree+1))
      for _ in range(self.ns)
    ]
    self.cost = 0
    self.constraints = []
    self._add_continuity_constraints()

  def add_cost(self):
    """
    Default cost
      - minimize highest order coefficients
    """
    for s in range(self.ns):
      self.cost += cp.sum_squares(self.spline_coeffs[s][:, -1])

  def _add_continuity_constraints(self):
    for s in range(self.ns-1):
      h = self.ts[s+1] - self.ts[s]

      for z in range(self.nflats):
        for sd in range(self.smoothness_degree+1):
          spline_end = self._eval_spline(h, sd, self.spline_coeffs[s][z])
          next_spline_start = self.spline_coeffs[s+1][z, sd] * factorial(sd)

          self.constraints += [spline_end == next_spline_start]

  def add_constraint(self, t, derivative_order, bounds, equality=False):
    """
    Add constraint to all flat outputs at derivative order
    """
    bounds = np.asarray(bounds).reshape(-1, 1)
    flats = cp.vstack(self.eval(t, derivative_order))
    self.constraints += [flats == bounds] if equality else [flats <= bounds]

  def _eval_spline(self, h, d, c):
    """
    h: float = time relative to start of spline
    d: int = derivative order
    c: cp.Variable = spline coefficients for a flat output
      - could be solved or unsolved depending on when this function is called
    """
    result = 0
    for pd in range(d, self.poly_degree+1):
      result += c[pd] * np.power(h, pd - d) * factorial(pd) / factorial(pd - d)
    return result

  def eval(self, t, derivative_order):
    """
    Evaluate flat outputs at a derivative order and time t
      - coefficients could be solved or unsolved depending on when this function is called
    """
    s = self.ts[self.ts <= t].argmax()
    if s >= self.ns:
      s = self.ns - 1
    h = t - self.ts[s]

    c = self.spline_coeffs[s] \
      if self.spline_coeffs[s].value is None \
      else self.spline_coeffs[s].value

    flats = []
    for z in range(self.nflats):
      flats.append(self._eval_spline(h, derivative_order, c[z]))

    return flats

  def solve(self, verbose=False):
    self.problem = cp.Problem(cp.Minimize(self.cost), self.constraints)
    self.problem.solve(verbose=verbose)

