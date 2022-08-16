import cv2
import numpy as np
from functools import partial

from gui.labeler import Labeler, LabelerAction


class Annotator:
    """
    Main gui class to annotate bounding box and to read the video
    """
    def __init__(self, cap, database):
        self._cap = cap
        self._database = database

        self._nb_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self._window_name = 'SAM'
        self._trackbar_name = 'time'
        self._ret = False
        self._frame = None
        self.init_window()

        # variables for drawing bb
        self._rect_drawn = False
        self._drawing_rect = False
        self._top_left = (0, 0)
        self._bottom_right = (0, 0)

        # variables to store annotations
        self._label = None
        self._obj_id = None
        self._labeler_action = None
        self._type_trajectory = None
        self._sequence_bb = []

        max_obj_per_video = 500
        np.random.seed(0)
        self.color_id = np.random.randint(low=0, high=255, size=(3, max_obj_per_video))

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
        callback = partial(Annotator.label_emitter, self)
        while cap.isOpened() and self._ret:
            frame_idx = cv2.getTrackbarPos(self._trackbar_name, self._window_name)
            if frame_idx != prev_frame_idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                self._ret, self._frame = cap.read()
            prev_frame_idx = frame_idx
            if self._ret:
                frame_to_show = self._frame.copy()
                for obj_id in self._database._database:
                    for sequence in self._database._database[obj_id][1]:
                        for chunk_idx in range(len(sequence.time_markers) - 1):
                            if sequence.time_markers[chunk_idx] < frame_idx < sequence.time_markers[chunk_idx + 1]:
                                cv2.rectangle(frame_to_show,
                                              sequence.bb[chunk_idx][0],
                                              sequence.bb[chunk_idx][1],
                                              self.color_id[:, obj_id].tolist(), 1)
                                cv2.rectangle(frame_to_show,
                                              sequence.bb[chunk_idx + 1][0],
                                              sequence.bb[chunk_idx + 1][1],
                                              self.color_id[:, obj_id].tolist(), 1)

                if self._drawing_rect:
                    cv2.rectangle(frame_to_show, self._top_left, self._bottom_right, (0, 255, 0), 2)
                if self._rect_drawn:
                    self._rect_drawn = False
                    show_rect = False
                    if len(self._sequence_bb) >= 1:
                        Labeler(callback, self._database)
                        self._sequence_bb.append((frame_idx, self._top_left, self._bottom_right, self._type_trajectory))

                        if self._labeler_action == LabelerAction.SAVE:
                            show_rect = True
                            self._database.add_sample(self._sequence_bb,
                                                      self._label,
                                                      self._obj_id)
                            self._sequence_bb = []
                        elif self._labeler_action == LabelerAction.CONTINUE:
                            show_rect = True
                        elif self._labeler_action == LabelerAction.CANCEL:
                            self._sequence_bb = []
                    else:
                        self._sequence_bb.append((frame_idx, self._top_left, self._bottom_right, None))
                        show_rect = True
                    if show_rect:
                        cv2.rectangle(frame_to_show, self._top_left, self._bottom_right, (0, 255, 0), 2)
                        self._frame = frame_to_show

                cv2.imshow(self._window_name, frame_to_show)
                c = cv2.waitKey(25)
                if c == ord('q'):
                    break
                elif c == ord('+') and frame_idx < self._nb_frames - 1:
                    cv2.setTrackbarPos(self._trackbar_name, self._window_name, frame_idx + 1)
                elif c == ord('-') and frame_idx > 1:
                    cv2.setTrackbarPos(self._trackbar_name, self._window_name, frame_idx - 1)

    def label_emitter(self, labeler_action, label, obj_id, type_trajectory):
        self._label = label
        self._obj_id = obj_id
        self._type_trajectory = type_trajectory
        self._labeler_action = labeler_action

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
            annotator._rect_drawn = True

