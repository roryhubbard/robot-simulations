import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from differentially_flat import DifferentialDriveTrajectory
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.meshcat_visualizer import ConnectMeshcatVisualizer
from pydrake.all import RigidTransform
from pydrake.math import RollPitchYaw


def set_pose(plant, plant_context, body_name,
             model_instance, p=[0, 0, 0], rpy=[0, 0, 0]):
  body = plant.GetBodyByName(body_name, plant.GetModelInstanceByName(model_instance)) \
    if isinstance(model_instance, str) \
    else plant.GetBodyByName(body_name, model_instance)
  plant.SetFreeBodyPose(plant_context, body,
                        RigidTransform(rpy=RollPitchYaw(rpy), p=p))


def get_box_vertices(plant, plant_context, box, l=1):
  """
  Assume unrotated cube with side length l
  """
  hl = l / 2
  x, y, z = plant.GetFreeBodyPose(plant_context, box).GetAsMatrix34()[:, -1]
  vertices = [
    [x + hl, y + hl],
    [x - hl, y + hl],
    [x - hl, y - hl],
    [x + hl, y - hl],
    [x + hl, y + hl],
  ]
  return vertices


def fill_pose(x=0, y=0, z=0, roll=0, pitch=0, yaw=0):
  return [x, y, z, roll, pitch, yaw]


def main():
  builder = DiagramBuilder()
  plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 1e-3)
  parser = Parser(plant)
  parser.AddModelFromFile('models/factory_bot.sdf', model_name='ego')
  parser.AddModelFromFile('models/factory_bot.sdf', model_name='other1')
  parser.AddModelFromFile('models/factory_bot.sdf', model_name='other2')
  boxes = [
    parser.AddModelFromFile('models/box.sdf', model_name='box1'),
    parser.AddModelFromFile('models/box.sdf', model_name='box2'),
    parser.AddModelFromFile('models/box.sdf', model_name='box3'),
    parser.AddModelFromFile('models/box.sdf', model_name='box4'),
    parser.AddModelFromFile('models/box.sdf', model_name='box5'),
    parser.AddModelFromFile('models/box.sdf', model_name='box6'),
    parser.AddModelFromFile('models/box.sdf', model_name='box7'),
    parser.AddModelFromFile('models/box.sdf', model_name='box8'),
  ]
  arms = [
    parser.AddModelFromFile('models/iiwa_arm.sdf', model_name='arm1'),
    parser.AddModelFromFile('models/iiwa_arm.sdf', model_name='arm2'),
  ]
  plant.Finalize()
  visualizer = ConnectMeshcatVisualizer(builder, scene_graph,
                                        zmq_url='tcp://127.0.0.1:6000')
  diagram = builder.Build()
  context = diagram.CreateDefaultContext()
  plant_context = plant.GetMyContextFromRoot(context)

  start_x = 0
  end_x = -3
  start_y = -3
  end_y = 3
  p0 = fill_pose(x=start_x, y=start_y)

  set_pose(plant, plant_context, 'chassis', 'ego', p=p0[:3], rpy=p0[3:])
  set_pose(plant, plant_context, 'chassis', 'other1', p=[-2, 4, 0], rpy=p0[3:])
  set_pose(plant, plant_context, 'chassis', 'other2', p=[-3, 3, 0], rpy=p0[3:])

  set_pose(plant, plant_context, 'box', boxes[0], p=[0, 0, .5])
  set_pose(plant, plant_context, 'box', boxes[1], p=[-1.1, 0, .5])
  set_pose(plant, plant_context, 'box', boxes[2], p=[-2.2, 0, .5])
  set_pose(plant, plant_context, 'box', boxes[3], p=[0, 1.1, .5])
  set_pose(plant, plant_context, 'box', boxes[4], p=[0, 2.2, .5])
  set_pose(plant, plant_context, 'box', boxes[5], p=[0, 3.3, .5])
  set_pose(plant, plant_context, 'box', boxes[6], p=[-2.6, 1.6, .5])
  set_pose(plant, plant_context, 'box', boxes[7], p=[-3.7, 1.6, .5])

  set_pose(plant, plant_context, 'iiwa_link_0', 'arm1', p=[0, 0, 1])
  set_pose(plant, plant_context, 'iiwa_link_0', 'arm2', p=[-2.6, 1.6, 1], rpy=[0, 0, np.pi])
  for arm in arms:
    plant.GetJointByName('iiwa_joint_2', arm).set_angle(plant_context, -np.pi/4)
    plant.GetJointByName('iiwa_joint_4', arm).set_angle(plant_context, -np.pi/4)
    plant.GetJointByName('iiwa_joint_6', arm).set_angle(plant_context, -np.pi/4)

  visualizer.load()
  diagram.Publish(context)

  N = 22
  t0 = 0
  tf = 10
  time_samples = np.linspace(t0, tf, N)
  n_flat_outputs = 2
  poly_degree = 3
  smoothness_degree = 2

  ddt = DifferentialDriveTrajectory(time_samples, n_flat_outputs,
                                    poly_degree, smoothness_degree)

  ddt.add_cost()
  ddt.add_constraint(t=0, derivative_order=0, bounds=[start_x, start_y], equality=True)
  ddt.add_constraint(t=0, derivative_order=1, bounds=[0, 0], equality=True)
  ddt.add_constraint(t=0, derivative_order=2, bounds=[0, 0], equality=True)
  ddt.add_constraint(t=tf, derivative_order=0, bounds=[end_x, end_y], equality=True)
  ddt.add_constraint(t=tf, derivative_order=1, bounds=[0, 0], equality=True)
  ddt.add_constraint(t=tf, derivative_order=2, bounds=[0, 0], equality=True)

  fig, ax = plt.subplots()

  ob_checks = np.linspace(t0, tf, N*2)
  for box in boxes:
    v = get_box_vertices(plant, plant_context, plant.GetBodyByName('box', box))
    ddt.add_obstacle(v, ob_checks, buffer=.2)
    ax.plot(*zip(*v))

  ob1 = np.array([
    [0, 1],
    [-10, 1],
    [-10, -1],
    [0, -1],
    [0, 1],
  ])
#  ddt.add_obstacle(ob1, ob_checks)

  ddt.solve()

  x_traj = []
  y_traj = []
  yaw_traj = []
  t_traj = np.linspace(t0, tf, N*2)
  for t in t_traj:
    flats = ddt.eval(t, 0)
    x_traj.append(flats[0])
    y_traj.append(flats[1])
    yaw_traj.append(ddt.recover_yaw(t, 0))

  visualizer.start_recording()
  for t in range(len(t_traj)):
    context.SetTime(t_traj[t])
    p = fill_pose(x=x_traj[t], y=y_traj[t], yaw=yaw_traj[t])
    set_pose(plant, plant_context, 'chassis', 'ego',
             p=p[:3], rpy=RollPitchYaw(p[3:]))
    diagram.Publish(context)
  visualizer.stop_recording()
  visualizer.publish_recording()

  ax.plot(x_traj, y_traj, '.')
  plt.gca().set_aspect('equal')
  #plt.show()
  #plt.close()


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

