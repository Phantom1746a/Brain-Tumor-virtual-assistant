from config import Roboflow_key
from roboflow import Roboflow
rf = Roboflow(api_key=Roboflow_key) # put your api key
project = rf.workspace("iotseecs").project("brain-tumor-yzzav")
version = project.version(1)
dataset = version.download("yolov11")