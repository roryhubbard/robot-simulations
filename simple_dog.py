import numpy as np
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.meshcat_visualizer import ConnectMeshcatVisualizer


def set_home(plant, context):
  plant.GetJointByName("hip_front").set_angle(context, -np.pi/4)
  plant.GetJointByName("hip_back").set_angle(context, np.pi/4)
  plant.GetJointByName("knee_front").set_angle(context, np.pi/2)
  plant.GetJointByName("knee_back").set_angle(context, -np.pi/2)


def main():
  builder = DiagramBuilder()
  plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 1e-3)
  parser = Parser(plant)
  simple_dog = parser.AddModelFromFile("models/simple_dog_2d.sdf")
  ground = parser.AddModelFromFile("models/little_dog/ground.urdf")
  plant.Finalize()
  visualizer = ConnectMeshcatVisualizer(builder, scene_graph,
                                        zmq_url="tcp://127.0.0.1:6000")
  diagram = builder.Build()
  context = diagram.CreateDefaultContext()
  plant_context = plant.GetMyContextFromRoot(context)
  set_home(plant, plant_context)
  visualizer.load()
  diagram.Publish(context)

  q0 = plant.GetPositions(plant_context)
  body_frame = plant.GetFrameByName("torso")

  prog = MathematicalProgram()        

  #visualizer.reset_recording()
  #visualizer.start_recording()
  #visualizer.publish_recording()


if __name__ == "__main__":
  main()

