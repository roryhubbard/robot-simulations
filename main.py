import numpy as np
import matplotlib.pyplot as plt
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.meshcat_visualizer import ConnectMeshcatVisualizer
from pydrake.all import Solve, LinearSystem, DirectTranscription, RigidTransform


def main():
  # Discrete-time approximation of the double integrator.
  dt = 0.01;
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
  sys = LinearSystem(A, B, C, D, dt)

  builder = DiagramBuilder()
  plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 1e-3)
  parser = Parser(plant)
  sphere = parser.AddModelFromFile('models/sphere.sdf')
  plant.Finalize()
  visualizer = ConnectMeshcatVisualizer(builder, scene_graph,
                                        zmq_url='tcp://127.0.0.1:6000')
  diagram = builder.Build()
  context = diagram.CreateDefaultContext()
  plant_context = plant.GetMyContextFromRoot(context)

  x0 = [-2., 0., -2., 0., 0., 0.]
  xf = [0., 0., 0., 0., 0., 0.]

  plant.SetFreeBodyPose(plant_context, plant.GetBodyByName('body'),
                        RigidTransform(x0[::2]))
  visualizer.load()
  diagram.Publish(context)

  N = 284
  prog = DirectTranscription(sys, sys.CreateDefaultContext(), N)
  prog.AddBoundingBoxConstraint(x0, x0, prog.initial_state())
  prog.AddBoundingBoxConstraint(xf, xf, prog.final_state())
  x = prog.state()
  u = prog.input()
  #prog.AddRunningCost(10*x[0]**2 + x[1]**2)
  #prog.AddRunningCost(u**2)
  prog.AddConstraintToAllKnotPoints(u[0] <= 1)
  prog.AddConstraintToAllKnotPoints(u[0] >= -1)
  prog.AddConstraintToAllKnotPoints(u[1] <= 1)
  prog.AddConstraintToAllKnotPoints(u[1] >= -1)

  result = Solve(prog)
  assert(result.is_success()), "Optimization failed"

  traj = prog.ReconstructStateTrajectory(result)
  t = traj.get_segment_times()
  x_values = traj.vector_values(t)
  x_sol = x_values[0, :]
  y_sol = x_values[2, :]
  z_sol = x_values[4, :]

  #plt.figure()
  #plt.plot(x_values[0,:], x_values[1,:])
  #plt.xlabel('x_sol')
  #plt.ylabel('qdot');
  #plt.show()
  #plt.close()

  visualizer.start_recording()
  for i in range(len(x_sol)):
    context.SetTime(t[i])
    plant.SetFreeBodyPose(plant_context, plant.GetBodyByName('body'),
                          RigidTransform([x_sol[i], y_sol[i], z_sol[i]]))
    diagram.Publish(context)
  visualizer.stop_recording()
  visualizer.publish_recording()


if __name__ == '__main__':
  main()

