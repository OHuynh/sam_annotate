import argparse
import cv2

from data import database
from gui.annotator import Annotator

parser = argparse.ArgumentParser()
parser.add_argument(
    "--video", type=str, help="The path to the video to annotate."
)

parser.add_argument(
    "--input-annot", type=str, help="The path to the previously generated json annotation to import."
)

parser.add_argument(
    "--output-path", type=str, help="This path will be used to store the annotation and the associate images.",
    default='./'
)

parser.add_argument(
    "--detector", type=str, help="The detector to use to interpolate annotations. Choice : [None, yolo5, yolo7]",
    choices=['', 'yolo5', 'yolo7'], default=''
)

parser.add_argument(
    "--path-yolo5-model", type=str, help="[If detector is Yolo5] This folder contains the yolo model hub.", default='./'
)

parser.add_argument(
    "--path-yolo7-model", type=str, help="[If detector is Yolo7] This folder contains the yolo model.", default='./'
)

parser.add_argument(
    "--path-yolo7-data", type=str, help="[If detector is Yolo7] This folder contains "
                                       "yolo txt files (output from detection).", default='./'
)

parser.add_argument(
    "--path-yolo7-project", type=str,
    help="[If detector is Yolo7] This folder contains the repository of the yolov7 project.", default='./'
)

parser.add_argument(
    "--execute-yolo7", type=str, help="[If detector is Yolo7] Indicate if YOLO is executed before open de app.",
    default="False"
)

if __name__ == "__main__":
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), 'Error reading the video'
    detector = None
    if args.detector == 'yolo7':
        from detector.yolo.yolov7 import load_yolo_model
        detector = load_yolo_model(args.path_yolo7_model,
                                   args.path_yolo7_data,
                                   args.video,
                                   args.execute_yolo7,
                                   args.path_yolo7_project)
    elif args.detector == 'yolo5':
        from detector.yolo.yolov5.yolov5 import YoloV5
        detector = YoloV5(cap, args.path_yolo5_model)

    database = database.Database(args.video, args.output_path, detector)

    database.load_json(args.input_annot)

    annotator = Annotator(cap, database)
    annotator.run()

    cap.release()
    cv2.destroyAllWindows()
