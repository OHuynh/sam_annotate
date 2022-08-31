import argparse
import cv2

from gui.annotator import Annotator
from gui.labelling import load_yolo_model
from data import database
parser = argparse.ArgumentParser()
parser.add_argument(
    "--video", type=str, help="The path to the video to annotate."
)

parser.add_argument(
    "--output-path", type=str, help="This path will be used to store the annotation and the associate images."
)

parser.add_argument(
    "--path-yolo-model", type=str, help="This folder contains the yolo model."
)

if __name__ == "__main__":
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), 'Error reading the video'

    database = database.Database(args.output_path)
    YOLO_model = load_yolo_model(args.path_yolo_model)
    annotator = Annotator(cap, database, YOLO_model)
    annotator.run()

    cap.release()
    cv2.destroyAllWindows()
