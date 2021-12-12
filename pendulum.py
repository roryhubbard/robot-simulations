import numpy as np
from pydrake.common import FindResourceOrThrow
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.meshcat_visualizer import ConnectMeshcatVisualizer


def run_pendulum_example(duration=1.):
  builder = DiagramBuilder()
  plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.)
  parser = Parser(plant)
  parser.AddModelFromFile(FindResourceOrThrow(
      "drake/examples/pendulum/Pendulum.urdf"))
  plant.Finalize()

  meshcat_vis = ConnectMeshcatVisualizer(builder, scene_graph, zmq_url="tcp://127.0.0.1:6000")

  diagram = builder.Build()
  simulator = Simulator(diagram)
  simulator.Initialize()
  simulator.set_target_realtime_rate(1.)

  # Fix the input port to zero.
  plant_context = diagram.GetMutableSubsystemContext(
      plant, simulator.get_mutable_context())
  plant.get_actuation_input_port().FixValue(
      plant_context, np.zeros(plant.num_actuators()))
  plant_context.SetContinuousState([0.5, 0.1])

  # Reset the recording (in case you are running this cell more than once).
  meshcat_vis.reset_recording()

  # Start recording and simulate.
  meshcat_vis.start_recording()
  simulator.AdvanceTo(duration)

  # Publish the recording to meshcat.
  meshcat_vis.publish_recording()


def main():
  run_pendulum_example()


if __name__ == "__main__":
  main()

