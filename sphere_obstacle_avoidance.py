import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.meshcat_visualizer import ConnectMeshcatVisualizer
from pydrake.all import RigidTransform


def make_system():
  dt = 0.1;
  A_tile = np.array([
    [1., dt],
    [0., 1.],
  ])
  A = np.kron(np.eye(3), A_tile)
  B = np.array([
    [0., 0.],
    [dt, 0.],
    [0., 0.],
    [0., dt],
    [0., 0.],
    [0., 0.],
  ])
  C = np.eye(6)
  D = np.zeros((6, 2))
  return A, B, C, D, dt


def drake_setup():
  builder = DiagramBuilder()
  plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 1e-3)
  parser = Parser(plant)
  parser.AddModelFromFile('models/sphere.sdf')
  plant.Finalize()
  visualizer = ConnectMeshcatVisualizer(builder, scene_graph,
                                        zmq_url='tcp://127.0.0.1:6000')
  diagram = builder.Build()
  context = diagram.CreateDefaultContext()
  plant_context = plant.GetMyContextFromRoot(context)
  return plant, visualizer, diagram, context, plant_context


def main():
  plant, visualizer, diagram, context, plant_context = drake_setup()
  A, B, C, D, dt = make_system()

  x0 = [-2., 0., -2., 0., 0., 0.]
  xN = [2., 0., 2., 0., 0., 0.]

  plant.SetFreeBodyPose(plant_context, plant.GetBodyByName('body'),
                        RigidTransform(x0[::2]))
  visualizer.load()
  diagram.Publish(context)

  # http://www.cs.cmu.edu/~zkolter/course/15-780-s14/mip.pdf
  u_limit = np.ones(2) * 10.
  null_control = np.zeros(2)
  obs_lbx = -1
  obs_ubx = 1
  obs_lby = -1
  obs_uby = 1
  bigM = 10
  N = 100

  x = cp.Variable((N, A.shape[0]))
  u = cp.Variable((N, B.shape[1]))
  z = cp.Variable((N, 4), boolean=True)

  constraints = []
  cost = 0
  for n in range(N-1):
    # obstacle avoidance using big M technique
    constraints += [x[n, 0] <= obs_lbx + z[n, 0] * bigM]
    constraints += [x[n, 0] >= obs_ubx - z[n, 1] * bigM]
    constraints += [x[n, 2] <= obs_lby + z[n, 2] * bigM]
    constraints += [x[n, 2] >= obs_uby - z[n, 3] * bigM]
    constraints += [cp.sum(z[n]) <= 3]

    # dynamics
    constraints += [x[n+1] == A @ x[n] + B @ u[n]]

    # control limits
    constraints += [u[n] <= u_limit]

    # smooth trajectory
    cost += cp.sum_squares(x[n+1] - x[n])

  # start and final states
  constraints += [x[0] == x0, x[N-1] == xN]
  # final control
  constraints += [u[N-1] == null_control]
  problem = cp.Problem(cp.Minimize(cost), constraints)
  problem.solve(solver=cp.GUROBI, verbose=True)

  visualizer.start_recording()
  for i in range(N):
    context.SetTime(dt*i)
    plant.SetFreeBodyPose(plant_context, plant.GetBodyByName('body'),
                          RigidTransform(x[i, ::2].value))

    diagram.Publish(context)
  visualizer.stop_recording()
  visualizer.publish_recording()

#  fig, ax = plt.subplots()
#  ax.plot(x[:, 0].value, x[:, 2].value)
#  obs_x = [obs_lbx, obs_lbx, obs_ubx, obs_ubx, obs_lbx]
#  obs_y = [obs_lby, obs_uby, obs_uby, obs_lby, obs_lby]
#  ax.plot(obs_x, obs_y)
#  plt.show()
#  plt.close()


if __name__ == '__main__':
  main()

