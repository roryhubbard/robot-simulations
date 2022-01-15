import numpy as np
from math import factorial
import cvxpy as cp
import matplotlib.pyplot as plt


class DifferentiallyFlatTrajectory:

  def __init__(self, ts, nflats, poly_degree, smoothness_degree):
    self.ts = ts
    self.nflats = nflats
    self.poly_degree = poly_degree
    self.smoothness_degree = smoothness_degree
    self.spline_coeffs = [
      cp.Variable((nflats, poly_degree+1))
      for _ in range(self.ts.size)
    ]
    self.cost = 0
    self.constraints = []
    self._add_continuity_constraints()

  def add_cost(self):
    """
    Default cost
      - minimize highest order elements
      - minimize distance between consecutive flat output states
    """
    for s in range(self.ts.size-1):
      self.cost += cp.sum_squares(self.spline_coeffs[s][:, -1])
     # spline = self.eval(s, 0)
     # next_spline = self.eval(s+1, 0)
     # for z in range(self.nflats):
     #   self.cost += cp.sum_squares(next_spline[z] - spline[z])

  def _add_continuity_constraints(self):
    for s in range(self.ts.size-1):
      h = self.ts[s+1] - self.ts[s]

      for z in range(self.nflats):
        for sd in range(self.smoothness_degree+1):
          spline_end = self._eval_spline(
            h, sd, self.spline_coeffs[s][z])
          next_spline_start = self.spline_coeffs[s+1][z, sd] \
            * np.math.factorial(sd)

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
    self.problem.solve(solver=cp.GUROBI, verbose=verbose)


class DifferentialDriveTrajectory(DifferentiallyFlatTrajectory):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def add_obstacle(self, vertices, checkpoints, bigM=10, buffer=0):
    """
    vertices: list(tuple(float)) = coordinates specifying vertices of obstacle
      - counterclockwise ordering and closed (first element == last elements)
    checkpoints: np.ndarray(float) = sample times to enforce as collision free
    """
    z = cp.Variable((checkpoints.size, len(vertices)-1), boolean=True)
    A = []
    b = []

    for i in range(len(vertices)-1):
      v1 = vertices[i]
      v2 = vertices[i+1]
      a = get_orthoganal_vector(v2 - v1)
      A.append(a)
      b.append(a @ v1)

    A = np.asarray(A)
    b = np.asarray(b)

    for t in checkpoints:
      flats = cp.vstack(self.eval(t, 0))
      bigM_rhs = cp.vstack(b + z[t] * bigM - buffer)
      self.constraints += [A @ flats <= bigM_rhs,
                           cp.sum(z[t]) <= len(vertices) - 2]

  def recover_yaw(self, t, derivative_order):
    """
    Recover yaw from flat outputs (x,y) at a derivative order and time t
    """
    if derivative_order > 0:
      # TODO: d/dx(arctan) = 1 / (1 + x^2)
      raise NotImplementedError
    x, y = self.eval(t, derivative_order+1)
    return np.arctan2(y, x)

  def recover_control_inputs(self, t, eps=1e-6):
    """
    Recover control inputs from flat outputs (x,y) at time t
    Return:
      - u1: longitudinal acceleration
      - u2: angular acceleration
    """
    x, y = self.eval(t, 4)
    yaw = self.recover_yaw(t, 0)
    long_accl = x / np.cos(yaw) if abs(np.cos(yaw)) > eps else y / np.sin(yaw)
    yaw_ddot = np.arctan2(y, x)
    return long_accl, yaw_ddot


def get_orthoganal_vector(v):
  R = np.array([
    [0, -1],
    [1, 0],
  ])
  return R @ v


def rotate(theta, vertices):
  R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)],
  ])
  return vertices @ R.T


def main():
  N = 5
  t0 = 0
  tf = 10
  time_samples = np.linspace(t0, tf, N)
  n_flat_outputs = 2
  poly_degree = 5
  smoothness_degree = 4

  ddt = DifferentialDriveTrajectory(time_samples, n_flat_outputs,
                                    poly_degree, smoothness_degree)

  ddt.add_cost()
  ddt.add_constraint(t=0, derivative_order=0, bounds=[-2, -2], equality=True)
  ddt.add_constraint(t=0, derivative_order=1, bounds=[0, 0], equality=True)
  ddt.add_constraint(t=0, derivative_order=2, bounds=[0, 0], equality=True)
  ddt.add_constraint(t=tf, derivative_order=0, bounds=[2, 2], equality=True)
  ddt.add_constraint(t=tf, derivative_order=1, bounds=[0, 0], equality=True)
  ddt.add_constraint(t=tf, derivative_order=2, bounds=[0, 0], equality=True)

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
  ddt.add_obstacle(square, checkpoints)

  ddt.solve(verbose=True)

  x = []
  y = []
  yaw = []
  long_accl = []
  ang_accl = []
  t_result = np.linspace(t0, tf, 100)
  for t in t_result:
    flats = ddt.eval(t, 0)
    x.append(flats[0])
    y.append(flats[1])
    yaw.append(ddt.recover_yaw(t, 0))
    u1, u2 = ddt.recover_control_inputs(t)
    long_accl.append(u1)
    ang_accl.append(u2)

  fig, ax = plt.subplots(ncols=2)
  ax[0].plot(x, y, '.')
  ax[0].plot(*zip(*square))
  ax[1].plot(yaw)
  plt.show()
  plt.close()


if __name__ == '__main__':
  plt.rcParams['figure.figsize'] = [16, 10]
  plt.rcParams['savefig.facecolor'] = 'black'
  plt.rcParams['figure.facecolor'] = 'black'
  plt.rcParams['figure.edgecolor'] = 'white'
  plt.rcParams['axes.facecolor'] = 'black'
  plt.rcParams['axes.edgecolor'] = 'white'
  plt.rcParams['axes.labelcolor'] = 'white'
  plt.rcParams['axes.titlecolor'] = 'white'
  plt.rcParams['xtick.color'] = 'white'
  plt.rcParams['ytick.color'] = 'white'
  plt.rcParams['text.color'] = 'white'
  plt.rcParams["figure.autolayout"] = True
  # plt.rcParams['legend.facecolor'] = 'white'
  main()

