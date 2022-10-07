import pandas as pd
from abc import abstractmethod

from detector.detector import Detector


class bounding_box:
    def __init__(self,frame_number,xmin,ymin,xmax,ymax,det_confidence,det_class,det_name):
        self.frame_number = frame_number
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.det_confidence = det_confidence
        self.det_class = det_class
        self.det_name = det_name

# list of all classes included in the model
list_of_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# mapping of the used labels with coco labels
#classes = ('Person', 'Autonomous Shuttle', 'Heat Shuttle', 'Car', 'Bicycle', 'Motorcycle', 'Bus', 'Truck')
label_coco_map = ['person', 'bus', 'bus', 'car', 'bicycle', 'motorcycle', 'bus', 'truck']


class YoloDetector(Detector):

    def interpolate(self, seq, label):

        coco_label = label_coco_map[label]
        coco_class_id = list_of_classes.index(coco_label)
        for sub_seq_idx, sub_seq in enumerate(seq.sub_sequence):
            initial_frame = seq.time_markers[sub_seq_idx]
            final_frame = seq.time_markers[sub_seq_idx + 1]
            initial_top_left = seq.bb[sub_seq_idx][0]
            initial_bottom_right = seq.bb[sub_seq_idx][1]
            final_top_left = seq.bb[sub_seq_idx + 1][0]
            final_bottom_right = seq.bb[sub_seq_idx + 1][1]

            initial_bb = bounding_box(initial_frame, initial_top_left[0], initial_top_left[1],
                                      initial_bottom_right[0], initial_bottom_right[1], 1.0,
                                      coco_class_id,
                                      coco_label)
            final_bb = bounding_box(final_frame, final_top_left[0], final_top_left[1],
                                    final_bottom_right[0], final_bottom_right[1], 1.0, coco_class_id,
                                    coco_label)
            df = self.track_object_with_YOLO(initial_frame,
                                             final_frame,
                                             initial_bb,
                                             final_bb)
            # fill the subsequence with dataframe
            for _, row in df.iterrows():
                sub_seq.insert_frame(time=row['frame'],
                                     top_left=(row['xmin'], row['ymin']),
                                     bottom_right=(row['xmax'], row['ymax']),
                                     type_traj=row['method'],
                                     idx=len(sub_seq.time_markers))

    @abstractmethod
    def track_object_with_YOLO(self, initial_frame, final_frame, initial_bb, final_bb):
        pass
