from ultralytics import YOLO

# Load a model
model = YOLO("best.pt")  # load a custom model

# Predict with the model
results = model("/root/autodl-tmp/dash-detr-master/datasets/GlobalWheat2020/images/inrae_1/f80a8b0d-a1f3-4fb3-b1b2-7fa3c9d2d0b4.png")  # predict on an image

# Access the results
for result in results:
    xywh = result.boxes.xywh  # center-x, center-y, width, height
    xywhn = result.boxes.xywhn  # normalized
    xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
    xyxyn = result.boxes.xyxyn  # normalized
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
    confs = result.boxes.conf  # confidence score of each box
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk