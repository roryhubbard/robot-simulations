import numpy as np
from math import factorial
import cvxpy as cp
import matplotlib.pyplot as plt


class DifferentiallyFlatTrajectory:
  """
  https://underactuated.mit.edu/trajopt.html
  http://www.cs.cmu.edu/~zkolter/course/15-780-s14/mip.pdf
  """

  def __init__(self, sample_times, n_flat_outputs, poly_degree, smoothness_degree):
    self.sample_times = sample_times
    self.n_flat_outputs = n_flat_outputs
    self.poly_degree = poly_degree
    self.smoothness_degree = smoothness_degree
    self.spline_coefficients = [
      cp.Variable((n_flat_outputs, poly_degree+1))
      for _ in range(self.sample_times.size)
    ]

    self.cost = 0
    self.constraints = []

    self.add_cost()
    self.add_continuity_constraints()

  def add_cost(self):
    for s in range(self.sample_times.size-1):
      self.cost += cp.sum_squares(self.spline_coefficients[s][:, -1])
      spline = self.eval(s, 0)
      next_spline = self.eval(s+1, 0)
      for z in range(self.n_flat_outputs):
        self.cost += cp.sum_squares(next_spline[z] - spline[z])

  def add_continuity_constraints(self):
    for s in range(self.sample_times.size-1):
      h = self.sample_times[s+1] - self.sample_times[s]

      for z in range(self.n_flat_outputs):
        for sd in range(self.smoothness_degree+1):
          spline_end = self.eval_spline(
            h, sd, self.spline_coefficients[s][z])
          next_spline_start = self.spline_coefficients[s+1][z, sd] \
            * np.math.factorial(sd)

          self.constraints += [spline_end == next_spline_start]

  def add_constraint(self, t, derivative_order, bounds,
                     equality=False, greater_than=False):
    """
    Add constraint to all flat outputs of specified derivative order
    """
    flat_outputs = self.eval(t, derivative_order)
    for z in range(len(flat_outputs)):
      self.add_single_constraint(flat_outputs[z], bounds[z], equality, greater_than)

  def add_single_constraint(self, lhs, rhs, equality=False, greater_than=False):
    """
    TODO: use canonical form
    """
    if equality:
      self.constraints += [lhs == rhs]
    else:
      if greater_than:
        self.constraints += [lhs >= rhs]
      else:
        self.constraints += [lhs <= rhs]

  def add_obstacle(self, vertices, checkpoints, bigM=10):
    """
    vertices: list(tuple(float)) = coordinates specifying vertices of obstacle
      - counterclockwise ordering and closed (first element == last elements)
    checkpoints: np.ndarray(float) = sample times to enforce as collision free
    TODO: use canonical form
    """
    eps = 1e-6
    b = cp.Variable((checkpoints.size, len(vertices)-1), boolean=True)

    for t in checkpoints:
      flat_outputs = self.eval(t, 0)

      for i in range(len(vertices)-1):
        v1 = vertices[i]
        v2 = vertices[i+1]
        y_delta = v2[1] - v1[1]
        x_delta = v2[0] - v1[0]
        greater_than = True
        if abs(y_delta) < eps:
          # horizontal
          rhs = v2[1]
          lhs_idx = 1
          if x_delta > 0:
            greater_than = False
        elif abs(x_delta) < eps:
          # vertical
          rhs = v2[0]
          lhs_idx = 0
          if y_delta < 0:
            greater_than = False
        else:
          # slanted
          m = y_delta / x_delta
          y_intercept = v2[1] - m * v2[0]
          rhs = m * flat_outputs[0] + y_intercept
          lhs_idx = 1
          theta = np.arctan2(y_delta, x_delta)
          if -np.pi / 2 < theta < np.pi / 2:
            greater_than = False

        if greater_than:
          bigM_rhs = rhs - b[t, i] * bigM
        else:
          bigM_rhs = rhs + b[t, i] * bigM

        lhs = flat_outputs[lhs_idx]
        self.add_single_constraint(lhs, bigM_rhs, greater_than=greater_than)
        self.add_single_constraint(cp.sum(b[t]), len(vertices)-2)

  def eval_spline(self, h, d, c):
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
    Evaluate flat outputs at a specified derivative order at time t
      - coefficients could be solved or unsolved depending on when this function is called
    """
    s = self.sample_times[self.sample_times <= t].argmax()
    h = t - self.sample_times[s]

    c = self.spline_coefficients[s] \
      if self.spline_coefficients[s].value is None \
      else self.spline_coefficients[s].value

    d = derivative_order
    flat_outputs = []
    for z in range(self.n_flat_outputs):
      flat_outputs.append(self.eval_spline(h, d, c[z]))
    
    return flat_outputs

  def solve(self):
    self.problem = cp.Problem(cp.Minimize(self.cost), self.constraints)
    self.problem.solve(solver=cp.GUROBI, verbose=True)


def rotate(theta, vertices):
  R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)],
  ])
  return (R @ vertices.T).T


def main():
  N = 22
  t0 = 0
  tf = 10
  sample_times = np.linspace(t0, tf, N)
  n_flat_outputs = 2
  poly_degree = 5
  smoothness_degree = 4

  dft = DifferentiallyFlatTrajectory(sample_times, n_flat_outputs,
                                     poly_degree, smoothness_degree)

  dft.add_constraint(t=0, derivative_order=0, bounds=[-2, -2], equality=True)
  dft.add_constraint(t=0, derivative_order=1, bounds=[0, 0], equality=True)
  dft.add_constraint(t=0, derivative_order=2, bounds=[0, 0], equality=True)
  dft.add_constraint(t=tf, derivative_order=0, bounds=[2, 2], equality=True)
  dft.add_constraint(t=tf, derivative_order=1, bounds=[0, 0], equality=True)
  dft.add_constraint(t=tf, derivative_order=2, bounds=[0, 0], equality=True)

  square = np.array([
    [1, 1],
    [-1, 1],
    [-1, -1],
    [1, -1],
    [1, 1],
  ])
  theta = np.pi / 4
  square = rotate(theta, square)

  checkpoints = np.linspace(t0, tf, 44)
  dft.add_obstacle(square, checkpoints)

  dft.solve()

  x = []
  y = []
  t_result = np.linspace(t0, tf, N)
  for t in t_result:
    flat_outputs = dft.eval(t, 0)
    x.append(flat_outputs[0])
    y.append(flat_outputs[1])

  fig, ax = plt.subplots()
  ax.plot(x, y, '.')
  ax.plot(*zip(*square))
  plt.show()
  plt.close()


if __name__ == '__main__':
  main()

