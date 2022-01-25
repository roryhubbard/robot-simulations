import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from piecewise_polynomial import PiecewisePolynomial


class DifferentialDriveTrajectory(PiecewisePolynomial):

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
  plt.rcParams['figure.figsize'] = [10, 8]
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

