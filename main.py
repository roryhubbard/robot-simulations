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

  simple_dog_file = "models/simple_dog_2d.sdf"
  simple_dog = parser.AddModelFromFile(simple_dog_file, model_name="simple_dog")

  ground_file = "models/little_dog/ground.urdf"
  ground = parser.AddModelFromFile(ground_file, model_name="ground")

  meshcat_vis = ConnectMeshcatVisualizer(builder, scene_graph,
                                         zmq_url="tcp://127.0.0.1:6000")

  plant.Finalize()
  diagram = builder.Build()

  # Create a new simulator (with its own implicitly created context).
  simulator = Simulator(diagram)
  # Set initial conditions.
  sim_plant_context = plant.GetMyContextFromRoot(simulator.get_mutable_context())

  plant.GetJointByName("knee_front").set_angle(sim_plant_context, .5)
  plant.GetJointByName("knee_back").set_angle(sim_plant_context, -.5)
  #plant.get_actuation_input_port(simple_dog).FixValue(sim_plant_context, np.zeros((4,1)))

  # Reset the recording (in case you are running this cell more than once).
  meshcat_vis.reset_recording()

  # Start recording and simulate.
  meshcat_vis.start_recording()
  simulator.AdvanceTo(1.0)

  # Publish the recording to meshcat.
  meshcat_vis.publish_recording()


if __name__ == "__main__":
  main()

