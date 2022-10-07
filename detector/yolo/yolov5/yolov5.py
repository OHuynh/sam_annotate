import torch
import cv2 as cv
import pandas as pd
import numpy as np
import sys
from shapely.geometry.polygon import Polygon

from detector.yolo.yolo import YoloDetector, bounding_box


class YoloV5(YoloDetector):
    def __init__(self, cap, path):
        super().__init__(cap)
        torch.hub.set_dir(path)
        YOLO_model = torch.hub.load('ultralytics/yolov5', 'yolov5m6')
        self.model = YOLO_model

    def track_object_with_YOLO(self, initial_frame, final_frame, initial_bb, final_bb):
        # dataframe that will store all boxes generated automatically between initial_bounding_box and final_bounding_box
        result_from_detection_columns = ["frame", "method", "percentage of intersection", "xmin", "ymin", "xmax",
                                         "ymax", "confidence", "class", "name"]
        result_from_detection = pd.DataFrame(columns=result_from_detection_columns)
        # Open the video file
        # cap = cv.VideoCapture(video_name)
        # save in the dataframe the initial_bounding_box, which is the first box of the sequence
        # set manually percentage of intersection = 1 because it is the first frame
        result_from_detection = save_in_result_from_detection(result_from_detection, 8, 1.0, initial_bb)
        # the first bouding box used as reference is the initial_bounding_box
        previous_bb = initial_bb
        # start yolo detection on frame two (one after initial_frame)
        # and stop one before final_frame
        for frame_number in list(range(initial_frame + 1, final_frame)):
            # set the frame to use
            self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
            # take the frame
            ret, frame = self.cap.read()
            # detect with yolo
            # output_from_detection is a dataframe that will contains all objects detected (with yolo) in frame_number
            output_from_detection = self.detect_single_frame_with_YOLO(frame, frame_number, result_from_detection)
            # inside output_from_detection look for the new box (representation of previous_bb in frame_number)
            df_new_box = find_new_box(previous_bb, output_from_detection, final_bb)
            ## too see old and new box
            # frame_copy = frame.copy()
            # frame_copy = cv.rectangle(frame_copy, (previous_bb.xmin,previous_bb.ymin), (previous_bb.xmax,previous_bb.ymax), (0, 255, 0), 1)
            # a = ((df_new_box.loc[min(list(df_new_box.index)),'xmin']), (df_new_box.loc[min(list(df_new_box.index)),'ymin']))
            # b = ((df_new_box.loc[min(list(df_new_box.index)),'xmax']), (df_new_box.loc[min(list(df_new_box.index)),'ymax']))
            # frame_copy = cv.rectangle(frame_copy, a, b, (0, 255, 255), 0)
            # windowsName = 'ventana de josh'
            # cv.namedWindow(windowsName)
            # cv.setMouseCallback(windowsName, mouseEvent)
            # cv.imshow(windowsName, frame_copy)
            # print(df_new_box)
            # choice = cv.waitKey(0)
            # add to df_new_box the column frame to match with result_from_detection
            column_frame = pd.DataFrame({'frame': [frame_number]}, index=list(df_new_box.index))
            df_new_box = pd.concat([column_frame, df_new_box], axis=1)
            # add df_new_box to result_from_detection
            result_from_detection = pd.concat([result_from_detection, df_new_box], axis=0, ignore_index=True)
            # create the new bouding box (here, it is supposed that only one box is saved in df_new_box
            previous_bb = bounding_box_from_df(df_new_box, list(df_new_box.index)[0])
        # return the sequence of created boxes (output is a dataframe)
        return result_from_detection

    def detect_single_frame_with_YOLO(self, frame, frame_number, df_model):
        """
        given a frame (and a frame_number), it consults if the frame was already passed through YOLO detector
        if possitive, then take the information from the self._data_from_yolo dataframe
        otherwise, pass frame through YOLO detector
        return:
            1) a dataframe with boxes found by YOLO detector
            2) a binary flag (True or False) indicating if frame had already been treated by YOLO detector or not
        """
        df = df_model[(df_model['frame']==frame_number)]
        # if yolo has been already executed over frame_number
        if len(df) > 0:
            # set that frame_number was already treated with yolo
            already_treated = True
            # output will be the detected objects in the corresponding frame
            results = df[["xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"]]
        # otherwise, process the frame (pass it through yolo)
        else:
            # set that frame_number has not been treated before with yolo
            already_treated = False
            df_columns = ["xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"]
            output = pd.DataFrame(columns=df_columns)
            # Detect Objects
            results = self.model(frame)
            results = results.pandas().xyxy[0]
            results['xmin'] = results['xmin'].astype(int)
            results['xmax'] = results['xmax'].astype(int)
            results['ymin'] = results['ymin'].astype(int)
            results['ymax'] = results['ymax'].astype(int)
        return results

def create_rectangle_from_bbclass(bounding_box):
    x1 = bounding_box.xmin
    x2 = bounding_box.xmax
    y1 = bounding_box.ymin
    y2 = bounding_box.ymax
    box = Polygon(np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1]]))
    return box

def create_rectangle_from_df(df, index_to_use):
    x1 = df.loc[index_to_use, "xmin"]
    x2 = df.loc[index_to_use, "xmax"]
    y1 = df.loc[index_to_use, "ymin"]
    y2 = df.loc[index_to_use, "ymax"]
    box = Polygon(np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1]]))
    return box


def save_in_result_from_detection(df, method_used, perc_of_intersec, bb):
    df.loc[len(df.index)] = [bb.frame_number,method_used, perc_of_intersec,
                             bb.xmin,bb.ymin,bb.xmax,bb.ymax,bb.det_confidence,bb.det_class,bb.det_name]
    return (df)


def bounding_box_from_df(df, index_to_use):
    bb = bounding_box(df.loc[index_to_use,"frame"], df.loc[index_to_use,"xmin"], df.loc[index_to_use,"ymin"],
                     df.loc[index_to_use, "xmax"], df.loc[index_to_use,"ymax"], df.loc[index_to_use,"confidence"],
                     df.loc[index_to_use, "class"], df.loc[index_to_use, "name"])
    return (bb)


# Function used to capture mouse events (clicks)
def mouseEvent(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv.EVENT_LBUTTONUP:
        mouseX,mouseY = x,y

## initial position for the mouse
mouseX=0
mouseY=0





def find_percentages_of_intersections(previous_bb, output_from_detection):
    # create the rectangle (box) concerning object_to_track
    box_object_to_track = create_rectangle_from_bbclass(previous_bb)
    # create the column to register intersection data
    extra_columns = ["percentage of intersection"]
    column_percentage_intersection = pd.DataFrame(columns=extra_columns)
    # add the column to dataframe output_from_detection
    output_from_detection = pd.concat([column_percentage_intersection, output_from_detection], axis=1)
    # for each row in output_from_detection, find the boxes that intersect with object_to_track
    for bb_index in list(output_from_detection.index):
        # create the rectangle (box)
        box_frame_two = create_rectangle_from_df(output_from_detection, bb_index)
        # register the intersection
        percentage_intersection = box_object_to_track.intersection(box_frame_two).area / box_object_to_track.area
        output_from_detection.loc[bb_index,"percentage of intersection"] = percentage_intersection
    return (output_from_detection)


def interpolate_box(previous_bb, final_bb, df):
    steps = final_bb.frame_number - previous_bb.frame_number
    interpolation_min = np.linspace((previous_bb.xmin, previous_bb.ymin), (final_bb.xmin, final_bb.ymin), steps)[1]
    interpolation_max = np.linspace((previous_bb.xmax, previous_bb.ymax), (final_bb.xmax, final_bb.ymax), steps)[1]
    df.loc[len(df.index)] = [1.0,int(interpolation_min[0]),int(interpolation_min[1]),int(interpolation_max[0]),int(interpolation_max[1]),
                             previous_bb.det_confidence, previous_bb.det_class, previous_bb.det_name]
    return (df)


def find_new_box(previous_bb,output_from_detection,final_bb):
    # set manually method; method_sed = 8 means that the box was created with yolo
    method_used = 8
    # set the threshold that will help to decide if two boxes intersect or not
    intersection_threshold = 0.5
    # compute the intersection between the previous box and all the objects detected in the frame
    df = find_percentages_of_intersections(previous_bb, output_from_detection)
    # take only the detected objects that have intersection with the previous box (intersection given by threshold)
    # and that have the same class than the previous box
    df = df[(df["percentage of intersection"] >= intersection_threshold) & (df["class"] == previous_bb.det_class)]
    # if there is not boxes then interpolate using previous_bb and final_bb
    if len(df) == 0:
        df = interpolate_box(previous_bb, final_bb, df)
        # set manually method; method_sed = 9 means interpolation
        method_used = 9
    # if there is more than one box select only the first one
    elif len(df) >= 2:
        df = df[df.index == min(list(df.index))]
    # add to df_new_box the column frame to match with result_from_detection
    column_method = pd.DataFrame({'method': [method_used]}, index=list(df.index))
    df = pd.concat([column_method, df], axis=1)
    return (df)



