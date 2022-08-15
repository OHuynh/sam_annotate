import cv2
from functools import partial

from gui.labeler import Labeler


class Annotator:
    """
    Main gui class to annotate bounding box and to read the video
    """
    def __init__(self, cap):
        self._cap = cap
        self._nb_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self._window_name = 'SAM'
        self._trackbar_name = 'time'
        self._ret = False
        self._frame = None
        self.init_window()

        # variables for drawing bb
        self._save_annot = False
        self._drawing_rect = False
        self._top_left = (0, 0)
        self._bottom_right = (0, 0)

        # variables to store annotations
        self._label = None
        self._obj_to_id = None

    def init_window(self):
        # init the window
        cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)
        self._ret, self._frame = self._cap.read()
        cv2.imshow(self._window_name, self._frame)
        cv2.createTrackbar(self._trackbar_name, self._window_name, 1, int(self._nb_frames), lambda _: None)
        cv2.setMouseCallback(self._window_name, Annotator._draw_bb, self)

    def run(self):
        cap = self._cap
        prev_frame_idx = 1
        while cap.isOpened() and self._ret:
            frame_idx = cv2.getTrackbarPos(self._trackbar_name, self._window_name)
            if frame_idx != prev_frame_idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                self._ret, self._frame = cap.read()
            prev_frame_idx = frame_idx
            if self._ret:
                frame_to_show = self._frame.copy()
                if self._drawing_rect:
                    cv2.rectangle(frame_to_show, self._top_left, self._bottom_right, (0, 255, 0), 2, 8)
                if self._save_annot:
                    cv2.rectangle(frame_to_show, self._top_left, self._bottom_right, (0, 255, 0), 2, 8)
                    self._save_annot = False
                    self._frame = frame_to_show
                    callback = partial(Annotator.label_emitter, self)
                    Labeler(callback)


                cv2.imshow(self._window_name, frame_to_show)
                c = cv2.waitKey(25)
                if c == ord('q'):
                    break
                elif c == ord('+') and frame_idx < self._nb_frames - 1:
                    cv2.setTrackbarPos(self._trackbar_name, self._window_name, frame_idx + 1)
                elif c == ord('-') and frame_idx > 1:
                    cv2.setTrackbarPos(self._trackbar_name, self._window_name, frame_idx - 1)

    def label_emitter(self, label, obj_id):
        self._label = label
        self._obj_to_id = obj_id

    @staticmethod
    def _draw_bb(action, x, y, flags, *userdata):
        annotator = userdata[0]
        if action == cv2.EVENT_LBUTTONDOWN:
            annotator._top_left = (x, y)
            annotator._bottom_right = (x, y)
            annotator._drawing_rect = True
        elif action == cv2.EVENT_MOUSEMOVE and annotator._drawing_rect:
            annotator._bottom_right = (x, y)
        elif action == cv2.EVENT_LBUTTONUP:
            annotator._bottom_right = (x, y)
            annotator._drawing_rect = False
            annotator._save_annot = True

