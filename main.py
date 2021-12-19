import numpy as np
import cvxpy as cp
from differentially_flat import DifferentialDriveTrajectory, rotate
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.meshcat_visualizer import ConnectMeshcatVisualizer
from pydrake.all import RigidTransform
from pydrake.math import RollPitchYaw


def drake_setup():
  builder = DiagramBuilder()
  plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 1e-3)
  parser = Parser(plant)
  parser.AddModelFromFile('models/factory_bot.sdf')
  plant.Finalize()
  visualizer = ConnectMeshcatVisualizer(builder, scene_graph,
                                        zmq_url='tcp://127.0.0.1:6000')
  diagram = builder.Build()
  context = diagram.CreateDefaultContext()
  plant_context = plant.GetMyContextFromRoot(context)
  return plant, visualizer, diagram, context, plant_context


def get_free_body_pose(x=0, y=0, z=0, roll=0, pitch=0, yaw=0):
  return [x, y, z, roll, pitch, yaw]

def main():
  plant, visualizer, diagram, context, plant_context = drake_setup()

  start_x = -2
  end_x = 2
  start_y = -2
  end_y = 2
  p0 = get_free_body_pose(x=start_x, y=start_y)

  plant.SetFreeBodyPose(plant_context, plant.GetBodyByName('chassis'),
                        RigidTransform(rpy=RollPitchYaw(p0[3:]), p=p0[:3]))
  visualizer.load()
  diagram.Publish(context)

  N = 22
  t0 = 0
  tf = 10
  time_samples = np.linspace(t0, tf, N)
  n_flat_outputs = 2
  poly_degree = 5
  smoothness_degree = 4

  ddt = DifferentialDriveTrajectory(time_samples, n_flat_outputs,
                                    poly_degree, smoothness_degree)

  ddt.add_cost()
  ddt.add_constraint(t=0, derivative_order=0, bounds=[start_x, start_y], equality=True)
  ddt.add_constraint(t=0, derivative_order=1, bounds=[0, 0], equality=True)
  ddt.add_constraint(t=0, derivative_order=2, bounds=[0, 0], equality=True)
  ddt.add_constraint(t=tf, derivative_order=0, bounds=[end_x, end_y], equality=True)
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
  #square = rotate(theta, square)

  checkpoints = np.linspace(t0, tf, 44)
  ddt.add_obstacle(square, checkpoints)

  ddt.solve()

  x_traj = []
  y_traj = []
  yaw_traj = []
  t_traj = np.linspace(t0, tf, 100)
  for t in t_traj:
    flats = ddt.eval(t, 0)
    x_traj.append(flats[0])
    y_traj.append(flats[1])
    yaw_traj.append(ddt.recover_yaw(t, 0))

  visualizer.start_recording()
  for t in range(len(t_traj)):
    context.SetTime(t_traj[t])
    p = get_free_body_pose(x=x_traj[t], y=y_traj[t], yaw=yaw_traj[t])
    plant.SetFreeBodyPose(plant_context, plant.GetBodyByName('chassis'),
                          RigidTransform(rpy=RollPitchYaw(p[3:]), p=p[:3]))
    diagram.Publish(context)
  visualizer.stop_recording()
  visualizer.publish_recording()


if __name__ == '__main__':
  main()

