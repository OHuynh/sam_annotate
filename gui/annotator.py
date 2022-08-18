import cv2
import numpy as np
from functools import partial

from utils.distinct_colors import COLORS
from utils.geometry import get_tl_br

from gui.labeler import Labeler, LabelerAction
from data.sequence_bound import SequenceBound


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
        self._mode_play = False
        self.init_window()

        # variables for drawing bb
        self._mode_rect_drawn = False
        self._mode_drawing_rect = False
        self._top_left = (0, 0)
        self._bottom_right = (0, 0)

        # variables for editing bb
        self._edit_pos = (0, 0)
        self._bb_edited = None
        self._mode_editing_rect = False
        self._mode_edit_at_click = False
        self._mode_rect_edited = False

        # variables to store annotations
        self._label = None
        self._obj_id = None
        self._labeler_action = None
        self._type_trajectory = None
        self._sequence_bb = []

        self.color_id = np.array(COLORS, dtype=np.uint8)
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.color_annotate = (0, 255, 0)

        self._boxes_interpolated = []

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
            if self._mode_play:
                frame_idx += 1
            if frame_idx != prev_frame_idx:
                cv2.setTrackbarPos(self._trackbar_name, self._window_name, frame_idx)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                self._ret, self._frame = cap.read()
            prev_frame_idx = frame_idx
            if self._ret:
                frame_to_show = self._frame.copy()

                # draw interpolated boxes
                for box in self._boxes_interpolated:
                    if box[0] == frame_idx:
                        cv2.rectangle(frame_to_show, box[1][0], box[1][1], (255, 0, 0), 2)

                chunks_displayed = []
                for obj_id in self._database.database:
                    color = self.color_id[obj_id, :].tolist()
                    label = f'{obj_id} {Labeler.classes[self._database.database[obj_id][0]]}'
                    for sequence in self._database.database[obj_id][1]:
                        frame_to_show, new_chunks_displayed = self.display_chunk_seq(sequence,
                                                                                     frame_idx,
                                                                                     frame_to_show,
                                                                                     color,
                                                                                     label)
                        chunks_displayed = new_chunks_displayed + chunks_displayed

                if len(self._sequence_bb) > 0:
                    # add boundaries rect to build a valid sequence to show
                    tmp_sequence_bb = self._sequence_bb.copy()
                    tmp_sequence_bb.insert(0,  (0, (0, 0), (0, 0), 0))
                    tmp_sequence_bb.append((self._nb_frames, (0, 0), (0, 0), 0))
                    tmp_sequence = SequenceBound(tmp_sequence_bb)
                    self.display_chunk_seq(tmp_sequence, frame_idx, frame_to_show, self.color_annotate, '')

                if self._mode_editing_rect:
                    if len(self._sequence_bb) == 0 and self._mode_edit_at_click:
                        nearest_point_clicked = []
                        for chunk in chunks_displayed:
                            top_left = np.array(chunk[0].bb[chunk[1]][0])
                            bottom_right = np.array(chunk[0].bb[chunk[1]][1])
                            pos = np.array(self._edit_pos)
                            if top_left[0] <= self._edit_pos[0] <= bottom_right[0] and \
                               top_left[1] <= self._edit_pos[1] <= bottom_right[1]:
                                # tl, tr, bl, br
                                nearest_point_clicked.append((np.linalg.norm(top_left - pos), 0, chunk))
                                nearest_point_clicked.append((np.linalg.norm([bottom_right[0] - pos[0],
                                                                              top_left[1] - pos[1]]), 1, chunk))
                                nearest_point_clicked.append((np.linalg.norm([top_left[0] - pos[0],
                                                                              bottom_right[1] - pos[1]]), 2, chunk))
                                nearest_point_clicked.append((np.linalg.norm(bottom_right - pos), 3, chunk))
                        # edit the closest point from where the user has clicked
                        nearest_point_clicked.sort(key=lambda pt: pt[0])
                        if len(nearest_point_clicked):
                            self._bb_edited = nearest_point_clicked[0][1:]

                    self._mode_edit_at_click = False
                    if self._bb_edited:
                        self._bb_edited[1][0].edit(self._bb_edited[0],
                                                   self._bb_edited[1][1],
                                                   self._edit_pos,
                                                   frame_idx)
                else:
                    self._bb_edited = None

                if self._mode_drawing_rect:
                    cv2.rectangle(frame_to_show, self._top_left, self._bottom_right, self.color_annotate, 2)

                if self._mode_rect_drawn:
                    self._mode_rect_drawn = False
                    show_rect = False
                    self._top_left, self._bottom_right = get_tl_br(self._top_left, self._bottom_right)

                    # check if another rect has been drawn at this frame (and overwrite it with this new one)
                    overwrite = False
                    for idx in range(len(self._sequence_bb)):
                        if self._sequence_bb[idx][0] == frame_idx:
                            overwrite = True
                            break
                    if overwrite:
                        self._sequence_bb.pop(idx)

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
                        elif self._labeler_action == LabelerAction.DISCARD:
                            self._sequence_bb = []
                        elif self._labeler_action == LabelerAction.CANCEL:
                            self._sequence_bb.pop(-1)

                    else:
                        self._sequence_bb.append((frame_idx, self._top_left, self._bottom_right, None))
                        show_rect = True
                    if show_rect:
                        cv2.rectangle(frame_to_show, self._top_left, self._bottom_right, self.color_annotate, 2)

                cv2.imshow(self._window_name, frame_to_show)
                c = cv2.waitKey(25)

                if c == ord('q'):
                    break
                elif c == ord('n') and len(self._sequence_bb):
                    if len(self._sequence_bb) == 1:
                        Labeler(callback, self._database)
                        sequence_bb = list(self._sequence_bb[0])
                        sequence_bb[3] = self._type_trajectory
                        self._sequence_bb[0] = tuple(sequence_bb)

                    self._database.add_sample(self._sequence_bb,
                                              self._label,
                                              self._obj_id)
                    self._sequence_bb = []
                elif c == ord('+') and frame_idx < self._nb_frames - 1:
                    cv2.setTrackbarPos(self._trackbar_name, self._window_name, frame_idx + 1)
                    if self._mode_editing_rect and self._bb_edited:
                        self._bb_edited[1][0].edit(self._bb_edited[0],
                                                   self._bb_edited[1][1],
                                                   self._edit_pos,
                                                   frame_idx)
                elif c == ord('-') and frame_idx > 1:
                    cv2.setTrackbarPos(self._trackbar_name, self._window_name, frame_idx - 1)
                    if self._mode_editing_rect and self._bb_edited:
                        self._bb_edited[1][0].edit(self._bb_edited[0],
                                                   self._bb_edited[1][1],
                                                   self._edit_pos,
                                                   frame_idx)
                elif c == ord('a'):
                    self._boxes_interpolated = self.compute_interpolation()
                elif c == ord(' '):
                    if self._mode_editing_rect or self._mode_drawing_rect:
                        print("Release editing/drawing mode to play the video.")
                    else:
                        self._mode_play = not self._mode_play

    def display_chunk_seq(self, sequence, frame_idx, frame_to_show, color, label):
        chunks_displayed = []
        for chunk_idx in range(len(sequence.time_markers) - 1):
            if sequence.time_markers[chunk_idx] <= frame_idx <= sequence.time_markers[chunk_idx + 1]:
                thickness = 2 if sequence.time_markers[chunk_idx] == frame_idx else 1
                cv2.rectangle(frame_to_show,
                              sequence.bb[chunk_idx][0],
                              sequence.bb[chunk_idx][1],
                              color, thickness)
                cv2.putText(frame_to_show,
                            label,
                            (sequence.bb[chunk_idx][0][0], sequence.bb[chunk_idx][0][1] + 15),
                            self.font, 1, color, 1)
                cv2.putText(frame_to_show,
                            f'{sequence.time_markers[chunk_idx]}',
                            (sequence.bb[chunk_idx][0][0], sequence.bb[chunk_idx][1][1] - 7),
                            self.font, 1, color, 1)

                thickness = 2 if sequence.time_markers[chunk_idx + 1] == frame_idx else 1
                cv2.rectangle(frame_to_show,
                              sequence.bb[chunk_idx + 1][0],
                              sequence.bb[chunk_idx + 1][1],
                              color, thickness)
                cv2.putText(frame_to_show,
                            label,
                            (sequence.bb[chunk_idx + 1][0][0], sequence.bb[chunk_idx + 1][0][1] + 15),
                            self.font, 1, color, 1)
                cv2.putText(frame_to_show,
                            f'{sequence.time_markers[chunk_idx + 1]}',
                            (sequence.bb[chunk_idx + 1][0][0], sequence.bb[chunk_idx + 1][1][1] - 7),
                            self.font, 1, color, 1)
                chunks_displayed.append((sequence, chunk_idx))
                chunks_displayed.append((sequence, chunk_idx + 1))
        return frame_to_show, chunks_displayed

    def label_emitter(self, labeler_action, label, obj_id, type_trajectory):
        self._label = label
        self._obj_id = obj_id
        self._type_trajectory = type_trajectory
        self._labeler_action = labeler_action

    @staticmethod
    def _draw_bb(action, x, y, flags, *userdata):
        annotator = userdata[0]

        if annotator._mode_play:
            return

        if not annotator._mode_editing_rect:
            if action == cv2.EVENT_LBUTTONDOWN:
                annotator._top_left = (x, y)
                annotator._bottom_right = (x, y)
                annotator._mode_drawing_rect = True
            elif action == cv2.EVENT_MOUSEMOVE and annotator._mode_drawing_rect:
                annotator._bottom_right = (x, y)
            elif action == cv2.EVENT_LBUTTONUP:
                annotator._bottom_right = (x, y)
                annotator._mode_drawing_rect = False
                annotator._mode_rect_drawn = True
        if not annotator._mode_drawing_rect:
            if action == cv2.EVENT_RBUTTONDOWN:
                annotator._edit_pos = (x, y)
                annotator._mode_edit_at_click = True
                annotator._mode_editing_rect = True
            elif action == cv2.EVENT_MOUSEMOVE and annotator._mode_editing_rect:
                annotator._edit_pos = (x, y)
            elif action == cv2.EVENT_RBUTTONUP:
                annotator._edit_pos = (x, y)
                annotator._mode_editing_rect = False
                annotator._mode_rect_edited = True

    def compute_interpolation(self):
        """
        TODO : @Josue implementation of the interpolation
        :return: list of tuples (frame, (top_left, bottom_right)) which will be displayed
        """
        # object containing the data
        # self._database.database
        # format : {key_id_0: [label_0, [[sequence_0], [sequence_1]]],
        #           key_id_1: ...}
        # sequence object of type SequenceBound

        # example linear interpol of first sequence of object 0 (between first and last frame)
        obj_id = 0
        seq_to_interpolate = 0
        assert obj_id in self._database.database, 'Object 0 not annotated yet'

        first_bb = self._database.database[obj_id][1][seq_to_interpolate].bb[0]
        last_bb = self._database.database[obj_id][1][seq_to_interpolate].bb[-1]

        first_time_marker = self._database.database[obj_id][1][seq_to_interpolate].time_markers[0]
        last_time_marker = self._database.database[obj_id][1][seq_to_interpolate].time_markers[-1]
        step = (np.array(last_bb) - np.array(first_bb)) / (last_time_marker - first_time_marker)
        interpolated_boxes = []
        for i in range(first_time_marker, last_time_marker):
            interpolated_boxes.append((i, (np.array(first_bb) + step * (i - first_time_marker)).astype(dtype=np.int32)))
        return interpolated_boxes
