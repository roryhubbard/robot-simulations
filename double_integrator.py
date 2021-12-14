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
  A = np.eye(2) + dt*np.mat('0 1; 0 0')
  B = dt*np.mat('0; 1')
  C = np.eye(2)
  D = np.zeros((2,1))
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

  x0 = [-2., 0.]
  xf = [0., 0.]

  plant.SetFreeBodyPose(plant_context, plant.GetBodyByName('body'),
                        RigidTransform([x0[0], 0, 0.1]))
  visualizer.load()
  diagram.Publish(context)

  N = 284
  prog = DirectTranscription(sys, sys.CreateDefaultContext(), N)
  prog.AddBoundingBoxConstraint(x0, x0, prog.initial_state())
  prog.AddBoundingBoxConstraint(xf, xf, prog.final_state())
  x = prog.state()
  u = prog.input()[0]
  #prog.AddRunningCost(10*x[0]**2 + x[1]**2)
  #prog.AddRunningCost(u**2)
  prog.AddConstraintToAllKnotPoints(u <= 1)
  prog.AddConstraintToAllKnotPoints(u >= -1)

  result = Solve(prog)
  assert(result.is_success()), "Optimization failed"

  x_sol = prog.ReconstructStateTrajectory(result)
  t = x_sol.get_segment_times()
  x_values = x_sol.vector_values(t)
  q = x_values[0,:]
  _qdot = x_values[1,:]

  #plt.figure()
  #plt.plot(x_values[0,:], x_values[1,:])
  #plt.xlabel('q')
  #plt.ylabel('qdot');
  #plt.show()
  #plt.close()

  visualizer.start_recording()
  for i, q in enumerate(q):
    context.SetTime(t[i])
    plant.SetFreeBodyPose(plant_context, plant.GetBodyByName('body'),
                          RigidTransform([q, 0, 0.1]))
    diagram.Publish(context)
  visualizer.stop_recording()
  visualizer.publish_recording()


if __name__ == '__main__':
  main()

