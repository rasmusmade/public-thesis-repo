from ultralytics import YOLO

# Load a model
model = YOLO("yolo11m-seg.yaml")  # build a new model from YAML
model = YOLO("yolo11m-seg.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11m-seg.yaml").load("yolo11m-seg.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="coco_human.yaml", epochs=100, imgsz=640, device=0, project="my_results", name="yolo11_medium_human_seg")
