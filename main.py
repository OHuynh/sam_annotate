import argparse
import cv2

from gui.annotator import Annotator

parser = argparse.ArgumentParser()
parser.add_argument(
    "--video", type=str, help="The path to the video to annotate."
)

if __name__ == "__main__":
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), 'Error reading the video'

    annotator = Annotator(cap)
    annotator.run()

    cap.release()
    cv2.destroyAllWindows()
