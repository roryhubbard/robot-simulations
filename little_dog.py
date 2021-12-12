from pathlib import Path
import numpy as np

from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.meshcat_visualizer import ConnectMeshcatVisualizer


def main():
  builder = DiagramBuilder()
  plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)

  parser = Parser(plant)

  little_dog_file = str(Path("LittleDog/LittleDog.urdf"))
  little_dog = parser.AddModelFromFile(little_dog_file, model_name="little_dog")
  #plant.WeldFrames(
  #    frame_on_parent_P=plant.world_frame(),
  #    frame_on_child_C=plant.GetFrameByName("iiwa_link_0", little_dog),
  #    X_PC=xyz_rpy_deg([0, -0.5, 0], [0, 0, 0]),
  #)

  ground_file = str(Path("LittleDog/ground.urdf"))
  ground = parser.AddModelFromFile(ground_file, model_name="ground")

  meshcat_vis = ConnectMeshcatVisualizer(builder, scene_graph, zmq_url="tcp://127.0.0.1:6000")

  plant.Finalize()
  diagram = builder.Build()

  # Create a new simulator (with its own implicitly created context).
  simulator = Simulator(diagram)
  # Set initial conditions.
  sim_plant_context = plant.GetMyContextFromRoot(simulator.get_mutable_context())
  plant.get_actuation_input_port(little_dog).FixValue(sim_plant_context, np.zeros((12,1)))
  plant.SetPositions(sim_plant_context, little_dog, [0.2, 0.4, 0, 0, 0, 0, 0])

  # Reset the recording (in case you are running this cell more than once).
  meshcat_vis.reset_recording()

  # Start recording and simulate.
  meshcat_vis.start_recording()
  simulator.AdvanceTo(2.0)

  # Publish the recording to meshcat.
  meshcat_vis.publish_recording()


if __name__ == "__main__":
  main()

