import argparse
import cv2

from gui.annotator import Annotator
from detector.labelling import load_yolo_model
from data import database
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
    "--path-yolo-model", type=str, help="This folder contains the yolo model.", default='./'
)

if __name__ == "__main__":
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), 'Error reading the video'

    YOLO_model = load_yolo_model(args.path_yolo_model)
    database = database.Database(args.video, args.output_path, YOLO_model)

    database.load_json(args.input_annot)

    annotator = Annotator(cap, database)
    annotator.run()

    cap.release()
    cv2.destroyAllWindows()
