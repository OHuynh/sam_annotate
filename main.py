import argparse
import cv2

from gui.annotator import Annotator
from data import database
parser = argparse.ArgumentParser()
parser.add_argument(
    "--video", type=str, help="The path to the video to annotate."
)

parser.add_argument(
    "--output-path", type=str, help="This path will be used to store the annotation and the associate images."
)

if __name__ == "__main__":
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), 'Error reading the video'

    database = database.Database(args.output_path)
    annotator = Annotator(cap, database)
    annotator.run()

    cap.release()
    cv2.destroyAllWindows()
