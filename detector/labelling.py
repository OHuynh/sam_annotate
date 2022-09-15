import torch
import cv2 as cv
import pandas as pd
import numpy as np
import math
from shapely.geometry.polygon import Polygon
import os, re, glob

from YOLOv7 import YOLOv7, utils


########## General functions
# extension of the labels' files
labels_extension = '.txt'
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

def find_files_with_pattern(path,pattern,extension):
    """
    This function returns a list of all files inside "path" that match "pattern" and "extension"
    i.e.  path/pattern.extension
    extension have to include the dot: extension=".mp4"
    the output format is: "file.extension" (no path)
    """
    files_with_pattern = []
    for file in glob.glob(path+pattern+extension):
        file = os.path.basename(file)
        files_with_pattern.append(file)
    return(files_with_pattern)

def find_filename_noExtension(filename):
    """
    Function used to find the name of a file (removing the extension)
    filename is composed as follows...
    input - filename = name.extension
    output - name
    """
    aux = re.split("[.]", filename)
    return aux[0]

def find_frameNumber(filename):
    """
    Function used to find the frame number
    filename is composed as follows...
    input - filename = CamX_Date_time_frameNumber
    output - frameNumber
    """
    aux = re.split("[_]", filename)
    return aux[-1]

def find_videoFileName(path_to_video):
    """
    Function used to find the name of a file (in this case a video) from a given path
    input: path_to_file/file_name.extension
    output: file_name
    dtype: str
    """
    aux = find_filename_noExtension(path_to_video)
    aux = re.split("[/]", aux)[-1]
    return aux

###########################

########### SAVE labels into dataframe

def loadDetectedObjectsFromYolo(pathToLabels, videoFileName, frame_width, frame_height):
    """
    return all the detected objects (boxes, output from YOLO) registered in pathToLabels (txt files) inside a dataframe
    df output: ["frame", "method", "percentage of intersection", "xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"]
    dtype: dataframe
    """
    # dataframe used to write txt files
    df_columns = ["frame", "method", "percentage of intersection", "xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"]
    df = pd.DataFrame(columns=df_columns)
    # row_number = to control the number of rows inserted in the dataframe df
    row_number = 0
    # Find the list of all txt files inside the folder "pathToLabels"
    path = pathToLabels
    pattern = videoFileName + "*"
    extension = labels_extension
    list_of_files = find_files_with_pattern(path, pattern, extension)
    # for each file, copy the detected objects from the txt file to the dataframe "df"
    for element in list_of_files:
        # find frame_number
        fileName_noExtension = find_filename_noExtension(element)
        frame_number = int(find_frameNumber(fileName_noExtension))
        # from the corresponding labelling (txt file)
        # extract and store all the detected objects inside the dataframe "df"
        # and update row_number (the number of inserted rows in df)
        df, row_number = storeObjectDetectionSingleFrame(pathToLabels, videoFileName, frame_number, df, row_number, frame_width, frame_height)
    return df

def storeObjectDetectionSingleFrame(pathToLabels, videoFileName, frame_number, df, row_number, frame_width, frame_height):
    """
    Save all the boxes found in a frame
    output 1) dataframe df: ["frame", "method", "percentage of intersection", "xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"]
    output 2) row_number: number of the actual row in df (required change: df[-1])
    """
    # get file txt used for labelling
    fileObject = open(pathToLabels + videoFileName + "_" + str(frame_number) + labels_extension, "r")
    # read all the detected objects saved inside the txt file
    while (True):
        # read next line
        line = fileObject.readline()
        # if line is empty, you are done with all lines in the file
        if not line:
            break
        # access the line
        aux = line.strip()
        # store object detected data
        df = storeObjectDetected(df, row_number, frame_number, aux, frame_width, frame_height)
        row_number += 1
    # close file txt
    fileObject.close
    # return row_number to know how many rows df has
    return df, row_number

def storeObjectDetected(df, row_number, frame_number, aux, frame_width, frame_height):
    """
    store a single box (object detected) from a given data (a line in a labels txt file)
    df output: ["frame", "method", "percentage of intersection", "xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"]
    dtype: dataframe
    """
    df.loc[row_number, "frame"] = int(frame_number)
    df.loc[row_number, "method"] = 'yolo'
    df.loc[row_number, "percentage of intersection"] = np.nan
    df.loc[row_number, "class"] = int(aux.split(" ")[0])
    df.loc[row_number, "name"] = list_of_classes[int(aux.split(" ")[0])]
    df.loc[row_number, "confidence"] = float(aux.split(" ")[5])
    # change xywh to xyxy
    x_center = float(aux.split(" ")[1]) * frame_width
    y_center = float(aux.split(" ")[2]) * frame_height
    width = float(aux.split(" ")[3]) * frame_width
    height = float(aux.split(" ")[4]) * frame_height
    xyxy = utils.xywh2xyxy(np.array([x_center,y_center,width,height]))
    df.loc[row_number, "xmin"] = int(xyxy[0])
    df.loc[row_number, "ymin"] = int(xyxy[1])
    df.loc[row_number, "xmax"] = int(xyxy[2])
    df.loc[row_number, "ymax"] = int(xyxy[3])
    return df

def xyxy2xywh(x):
    """
    Convert bounding box (xmin, ymin, xmax, ymax) to bounding box (x, y, w, h)
    """
    y = np.copy(x)
    y[..., 0] = (x[..., 2] - x[..., 0]) / 2 + x[..., 0]
    y[..., 1] = (x[..., 3] - x[..., 1]) / 2 + x[..., 1]
    y[..., 2] = (x[..., 2] - x[..., 0])
    y[..., 3] = (x[..., 3] - x[..., 1])
    return y

###########################


################ For YOLO inference
def load_yolo_model(path_to_model, path_to_yolo_data, path_to_video, execute_yolo):
    """
    Load the model (YOLO model) and a dataframe containing all the detected objects (boxes per frame)
    from YOLO execution (if any)
    output: dataframe_yolo_data, YOLO_model
    """
    # take only the file name
    videoFileName = find_videoFileName(path_to_video)
    # upload model
    YOLO_model = YOLOv7(path_to_model, conf_thres=0.3, iou_thres=0.5)
    print(f'model uploaded from {path_to_model}')
    # execute YOLO detector (if required)
    # for this, it is required to download yolo project
    if execute_yolo == 'True':
        python_env = 'C:/Users/josue.rivera/AppData/Local/Programs/Python/Python39/python.exe'
        path_to_detect = 'D:/SAM/yolov7-master/detect.py'
        weights = 'D:/SAM/yolov7-master/yolov7.pt'
        device = 0
        file_source = path_to_video
        project_path = path_to_yolo_data
        execute_YOLO_in_cmd(python_env, path_to_detect, file_source, weights, device, project_path, videoFileName)
    # check the size of images in the video (to denormalize images)
    # this must be changed into a dynamic form
    frame_width = 1280
    frame_height = 720
    # upload yolo data (if exist)
    path = path_to_yolo_data + '/' + videoFileName + '/' + 'labels' + '/'
    print(f'uploading data from previous YOLO executions from {path}...')
    YOLO_data = loadDetectedObjectsFromYolo(path, videoFileName, frame_width, frame_height)
    print(f'{len(YOLO_data)} rows were uploaded')
    return YOLO_data, YOLO_model


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


def detect_single_frame_with_YOLO(frame, frame_number, df_model, model):
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
        output = df[["xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"]]
    # otherwise, process the frame (pass it through yolo)
    else:
        # set that frame_number has not been treated before with yolo
        already_treated = False
        df_columns = ["xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"]
        output = pd.DataFrame(columns=df_columns)
        # Detect Objects
        boxes, scores, class_ids = model(frame)
        for i in range(len(class_ids)):  # detections per image
            # Write results
            output.loc[len(output.index)] = [int(boxes[i][0]),
                                             int(boxes[i][1]),
                                             int(boxes[i][2]),
                                             int(boxes[i][3]),
                                             scores[i],
                                             int(class_ids[i]),
                                             list_of_classes[int(class_ids[i])]]
    return output, already_treated


def save_tracked_object(df, method_used, perc_of_intersec, bb):
    df.loc[len(df.index)] = [bb.frame_number,method_used, perc_of_intersec,
                             bb.xmin,bb.ymin,bb.xmax,bb.ymax,bb.det_confidence,bb.det_class,bb.det_name]
    return df


def bounding_box_from_df(df, index_to_use):
    bb = bounding_box(df.loc[index_to_use,"frame"], df.loc[index_to_use,"xmin"], df.loc[index_to_use,"ymin"],
                     df.loc[index_to_use, "xmax"], df.loc[index_to_use,"ymax"], df.loc[index_to_use,"confidence"],
                     df.loc[index_to_use, "class"], df.loc[index_to_use, "name"])
    return bb



def track_object_with_YOLO(cap, initial_frame, final_frame, initial_bb, final_bb, data_from_yolo, YOLO_model):
    """
    Given an initial and final frame, it returns:
     1) data from YOLO detector. This data is the result of passing frames (initial+1 to final-1) through YOLO detector
     2) data corresponding to object marked with the boxes. This data is found inside YOLO detector results
    inputs: cap: video
            initial frame (number)
            final frame (number)
            initial_bb (class bouding box)
            final_bb (class bouding box)
            data_from_yolo - current version of the dataframe containing results from YOLO detector
                             (at the end of this function, this dataframe is replaced with the first output -
                             data from YOLO detector)
            YOLO_model - model used to predictions (object detection)
    """
    # structure of the output dataframes
    data_from_yolo_columns = ["frame", "method", "percentage of intersection", "xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"]
    # dataframe that will contain results from yolo detection (all objects detected)
    #data_from_yolo = pd.DataFrame(columns=data_from_yolo_columns)
    # dataframe that will contain results concerning the tracked object
    data_tracked_object = pd.DataFrame(columns=data_from_yolo_columns)
    # Open the video file
    # save in the tracked_object dataframe the initial_bounding_box, which is the first box of the sequence
    # set manually percentage of intersection = 1 because it is the first frame (set manually)
    data_tracked_object = save_tracked_object(data_tracked_object,'manual',1.0,initial_bb)
    # the first bouding box used as reference is the initial_bounding_box
    previous_bb = initial_bb
    # start yolo detection on frame two (one after initial_frame)
    # and stop one before final_frame
    for frame_number in list(range(initial_frame + 1, final_frame)):
        # set the frame to use
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
        # take the frame
        ret, frame = cap.read()
        # detect with yolo
        # output_from_detection is a dataframe that will contains all objects detected (with yolo) in frame_number
        # data_already_treated is a binary flag (False / True) that indicates if the frame had been already passed through yolo
        output_from_detection, data_already_treated = detect_single_frame_with_YOLO(frame, frame_number, data_from_yolo, YOLO_model)
        # if frame has not been treated before, then store data from yolo in dataframe data_from_yolo
        if not data_already_treated:
            column_frame = pd.DataFrame({'frame': [frame_number]*len(output_from_detection)}, index=list(output_from_detection.index))
            column_method = pd.DataFrame({'method': ['yolo']*len(output_from_detection)}, index=list(output_from_detection.index))
            column_poi = pd.DataFrame({'percentage of intersection': [np.nan] * len(output_from_detection)}, index=list(output_from_detection.index))
            df = pd.concat([column_frame, column_method, column_poi, output_from_detection], axis=1)
            data_from_yolo = pd.concat([data_from_yolo, df], axis=0, ignore_index=True)
        # inside output_from_detection look for the new box (look for tracked object in frame_number)
        df_new_box = find_new_box(previous_bb,output_from_detection,final_bb)
        # add to df_new_box the column frame to match with result_from_detection
        column_frame = pd.DataFrame({'frame': [frame_number]}, index=list(df_new_box.index))
        df_new_box = pd.concat([column_frame, df_new_box], axis=1)
        # add df_new_box to result_from_detection
        data_tracked_object = pd.concat([data_tracked_object, df_new_box], axis=0, ignore_index=True)
        # create the new bouding box (here, it is supposed that only one box is saved in df_new_box
        previous_bb = bounding_box_from_df(df_new_box, list(df_new_box.index)[0])
    # add the final frame (given manually as parameter)
    data_tracked_object = save_tracked_object(data_tracked_object, 'manual', 1.0, final_bb)
    # return the sequence of created boxes (output is a dataframe)
    return data_from_yolo, data_tracked_object



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
    return output_from_detection


def interpolate_box(previous_bb, final_bb, df):
    steps = final_bb.frame_number - previous_bb.frame_number
    interpolation_min = np.linspace((previous_bb.xmin, previous_bb.ymin), (final_bb.xmin, final_bb.ymin), steps)[1]
    interpolation_max = np.linspace((previous_bb.xmax, previous_bb.ymax), (final_bb.xmax, final_bb.ymax), steps)[1]
    df.loc[len(df.index)] = [1.0,int(interpolation_min[0]),int(interpolation_min[1]),int(interpolation_max[0]),int(interpolation_max[1]),
                             previous_bb.det_confidence, previous_bb.det_class, previous_bb.det_name]
    return df


def find_new_box(previous_bb,output_from_detection,final_bb):
    # set manually method; method_sed = 8 means that the box was created with yolo
    method_used = 'yolo'
    # set the threshold that will help to decide if two boxes intersect or not
    intersection_threshold = 0.4
    # compute the intersection between the previous box and all the objects detected in the frame
    df = find_percentages_of_intersections(previous_bb, output_from_detection)
    # take only the detected objects that have intersection with the previous box (intersection given by threshold)
    # and that have the same class than the previous box
    df = df[(df["percentage of intersection"] >= intersection_threshold) & (df["class"] == previous_bb.det_class)]
    # if there is not boxes then interpolate using previous_bb and final_bb
    if len(df) == 0:
        df = interpolate_box(previous_bb, final_bb, df)
        # set manually method interpolation
        method_used = 'interpolation'
    # if there is more than one box, then select the one with higher percentage of intersection
    elif len(df) >= 2:
        df = df[df["percentage of intersection"] == df["percentage of intersection"].max()]
        # ensure that there is only one single row
        df = df[df.index == min(list(df.index))]
    # add to df_new_box the column frame to match with result_from_detection
    column_method = pd.DataFrame({'method': [method_used]}, index=list(df.index))
    df = pd.concat([column_method, df], axis=1)
    return df


def execute_YOLO_in_cmd(python_env, path_to_detect, file_source, weights, device, project_path, name):
    """
    Function used to execute YOLO outside this app
    imputs:
        1) python environment
        2) path to detect.py file
        3) file to pass throughout YOLO
        4) weights file (file pt, including path)
        5) device, cuda:0, cpu:1,2,...
        6) path to save results (txt files)
        7) name of the folder that will contain all txt files
    example:
    C:/Users/josue.rivera/AppData/Local/Programs/Python/Python39/python.exe "D:/SAM/yolov7-master/detect.py"
    --source "D:/SAM/Data_to_label/Cam3_20210614_183955.mp4" --weights D:/SAM/yolov7-master/yolov7.pt
    --save-txt --device 0 --nosave --save-txt --project "D:/SAM/yolov7-master" --name "Cam3_20210614_183955"
    output (txt files) will be saved inside 6)/7)/labels
    return: full path to output
    """
    print(f"executing YOLO detection for file {file_source}...")
    # if yolo was already executed, then don't executed again
    yolo_files = find_files_with_pattern(project_path + '/' + name + '/' + 'labels' + '/', pattern='*', extension='.txt')
    if not len(yolo_files):
        # create the line to execute
        line = python_env + ' ' + '"' + path_to_detect + '"' +\
               ' --source ' + '"' + file_source + '"' + \
               ' --weights ' + '"' + weights + '"' + \
               ' --save-txt ' + \
               ' --save-conf ' + \
               ' --device ' + str(device) + \
               ' --nosave --save-txt' + \
               ' --project ' + '"' + project_path + '"' + \
               ' --name ' + '"' + name + '"'
        # execute the line in cmd
        os.system('cmd /c "{}"'.format(line))
        print(f'YOLO results were saved in {project_path}/{name}/labels')
    else:
        print(f'YOLO results already exist in {project_path}/{name}/labels')





















