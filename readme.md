# Video annotation tool for SAM project

A tool to quickly annotate bounding box using a semi-automated process. The user has to annotate the first and the last 
frame of a sequence, then the intermediate bounding boxes are interpolated using a pre-trained detector (like YOLO)
and a specified trajectory of the object (eg. linear).

