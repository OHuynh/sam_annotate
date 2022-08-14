import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument(
    "--video", type=str, help="The path to the video to annotate."
)

if __name__ == "__main__":
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), 'Error reading the video'
    nb_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    window_name = 'SAM'
    trackbar_name = 'time'
    # init the window
    frame_idx = prev_frame_idx = 1
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    ret, frame = cap.read()
    cv2.imshow(window_name, frame)
    cv2.createTrackbar(trackbar_name, window_name, frame_idx, int(nb_frames), lambda _: None)

    while cap.isOpened() and ret:
        frame_idx = cv2.getTrackbarPos(trackbar_name, window_name)
        if frame_idx != prev_frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
        prev_frame_idx = frame_idx
        if ret:
            cv2.imshow(window_name, frame)
            c = cv2.waitKey(25)
            if c == ord('q'):
                break
            elif c == ord('+') and frame_idx < nb_frames - 1:
                cv2.setTrackbarPos(trackbar_name, window_name, frame_idx + 1)
            elif c == ord('-') and frame_idx > 1:
                cv2.setTrackbarPos(trackbar_name, window_name, frame_idx - 1)

    cap.release()
    cv2.destroyAllWindows()
