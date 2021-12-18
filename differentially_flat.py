import numpy as np
from math import factorial
import cvxpy as cp
import matplotlib.pyplot as plt


class DifferentiallyFlatTrajectory:

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
    # minimize highest order terms
    for s in range(self.sample_times.size):
      self.cost += cp.sum_squares(self.spline_coefficients[s][:, -1])

  def add_continuity_constraints(self):
    for s in range(self.sample_times.size-1):
      h = self.sample_times[s+1] - self.sample_times[s]

      for z in range(self.n_flat_outputs):
        for sd in range(self.smoothness_degree+1):
          spline_end = self.eval_spline_derivative(
            h, sd, self.spline_coefficients[s][z])
          next_spline_start = self.spline_coefficients[s+1][z, sd] \
            * np.math.factorial(sd)
          self.constraints += [spline_end == next_spline_start]

  def add_equality_constraint(self, t, derivative_order, vals):
    flat_outputs = self.eval(t, derivative_order)
    for z in range(self.n_flat_outputs):
      self.constraints += [flat_outputs[z] == vals[z]]

  def eval_spline_derivative(self, h, d, c):
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
    Evaluate flat outputs at a given derivative order at time t
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
      flat_outputs.append(self.eval_spline_derivative(h, d, c[z]))
    
    return flat_outputs

  def solve(self):
    self.problem = cp.Problem(cp.Minimize(self.cost), self.constraints)
    self.problem.solve(solver=cp.GUROBI, verbose=True)


def main():
  N = 11
  t0 = 0
  tf = 10
  sample_times = np.linspace(t0, tf, N)
  n_flat_outputs = 2
  poly_degree = 5
  smoothness_degree = 4

  dft = DifferentiallyFlatTrajectory(sample_times, n_flat_outputs,
                                     poly_degree, smoothness_degree)

  dft.add_equality_constraint(t=0, derivative_order=0, vals=[-2, -2])
  dft.add_equality_constraint(t=0, derivative_order=1, vals=[0, 0])
  dft.add_equality_constraint(t=0, derivative_order=2, vals=[0, 0])
  dft.add_equality_constraint(t=tf, derivative_order=0, vals=[2, 2])
  dft.add_equality_constraint(t=tf, derivative_order=1, vals=[0, 0])
  dft.add_equality_constraint(t=tf, derivative_order=2, vals=[0, 0])

  dft.solve()

  x = []
  y = []
  t_result = np.linspace(t0, tf, 100)
  for t in t_result:
    flat_outputs = dft.eval(t, 0)
    x.append(flat_outputs[0])
    y.append(flat_outputs[1])

  fig, ax = plt.subplots()
  ax.plot(x, y)
  plt.show()
  plt.close()


if __name__ == '__main__':
  main()

